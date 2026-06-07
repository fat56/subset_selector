#!/usr/bin/env python3
"""Cache hard subsets, compute native labels, and train the Stage 2.0 readout."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vggt_omega_selector.backbones.vggt_omega import VGGTOmegaIntegration  # noqa: E402
from vggt_omega_selector.readouts.training import HardLabelTrainConfig, train_hardlabel_readout  # noqa: E402


PRIMARY_LABEL_METRICS = ("pose_rotation_mean_deg", "pointmap_rmse_norm", "depth_log_rmse")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 2.0 hard-label readout experiment.")
    parser.add_argument(
        "--manifest",
        default="docs/experiments/0003_stage2_readout_calibration/hardlabel100_manifest.json",
    )
    parser.add_argument(
        "--cache-root",
        default="caches/vggt_omega/0003_stage2_readout_calibration/hardlabel100_full100_80_images512",
    )
    parser.add_argument(
        "--run-dir",
        default="runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80",
    )
    parser.add_argument("--checkpoint", default="512")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--cache-devices", default="cuda:0,cuda:1")
    parser.add_argument("--train-devices", default="cuda:0,cuda:1")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pairs-per-scene", type=int, default=48)
    parser.add_argument("--max-pixels-per-image", type=int, default=1024)
    parser.add_argument("--max-pointmap-points", type=int, default=60000)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args(argv)

    manifest_path = resolve(args.manifest)
    cache_root = resolve(args.cache_root)
    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs, records = build_cache_jobs(manifest, cache_root, run_dir, args.max_scenes)
    (run_dir / "cache_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    (run_dir / "cache_records.json").write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")

    if not args.skip_cache:
        missing_jobs = [job for job in jobs if args.force_cache or not cache_ready(Path(job["output_dir"]))]
        if missing_jobs:
            run_cache_jobs_parallel(
                jobs=missing_jobs,
                run_dir=run_dir,
                checkpoint=args.checkpoint,
                image_resolution=args.image_resolution,
                mode=args.mode,
                devices=[device.strip() for device in args.cache_devices.split(",") if device.strip()],
                force=args.force_cache,
            )
        else:
            print(json.dumps({"event": "cache_skip", "reason": "all jobs already cached", "jobs": len(jobs)}), flush=True)

    validate_cache_records(records)
    if args.cache_only:
        print(json.dumps({"event": "cache_only_done", "records": len(records)}, indent=2), flush=True)
        return 0

    metrics_csv = run_dir / "hardlabel_native_metrics.csv"
    labels_csv = run_dir / "hardlabel_train_labels.csv"
    if not args.skip_labels or not metrics_csv.is_file() or not labels_csv.is_file():
        metric_rows = compute_native_metrics(
            records=records,
            max_pixels_per_image=args.max_pixels_per_image,
            max_pointmap_points=args.max_pointmap_points,
            epsilon=args.epsilon,
        )
        label_rows = build_training_labels(metric_rows, records)
        write_csv(metrics_csv, metric_rows)
        write_csv(labels_csv, label_rows)
        write_label_summary(run_dir / "hardlabel_summary.json", metric_rows, label_rows)
    else:
        print(json.dumps({"event": "labels_skip", "reason": "existing labels", "labels": str(labels_csv)}), flush=True)

    if args.labels_only:
        print(json.dumps({"event": "labels_only_done", "labels": str(labels_csv)}, indent=2), flush=True)
        return 0

    config = HardLabelTrainConfig(
        labels_csv=labels_csv,
        run_dir=run_dir,
        val_metrics_csv=resolve(
            "docs/experiments/0002_ltm30_pose_depth_validation/native_geometry/ltm30_subset_native_consistency.csv"
        ),
        val_cache_root=resolve("caches/vggt_omega/0002_ltm30_pose_depth_validation/native_geometry_images512"),
        devices=tuple(device.strip() for device in args.train_devices.split(",") if device.strip()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        pairs_per_scene=args.pairs_per_scene,
    )
    train_hardlabel_readout(config)
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def safe_scene_id(scene_key: str) -> str:
    return scene_key.replace("/", "__")


def build_cache_jobs(
    manifest: dict[str, Any],
    cache_root: Path,
    run_dir: Path,
    max_scenes: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    scenes = manifest["scenes"][:max_scenes] if max_scenes else manifest["scenes"]
    for scene in scenes:
        scene_id = scene.get("scene_id") or safe_scene_id(scene["scene_key"])
        frame_by_id = {frame["frame_id"]: frame for frame in scene["frames"]}
        for method, frame_ids in scene["splits"].items():
            image_paths = []
            for frame_id in frame_ids:
                frame = frame_by_id[frame_id]
                image_paths.append(str((PROJECT_ROOT / frame["image_path"]).resolve()))
            image_list = run_dir / "image_lists" / scene_id / f"{method}.txt"
            image_list.parent.mkdir(parents=True, exist_ok=True)
            image_list.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
            output_dir = cache_root / scene_id / method
            role = "reference" if method == "full" else "subset"
            job = {
                "id": f"{scene_id}/{method}",
                "scene_id": scene_id,
                "scene_key": scene["scene_key"],
                "dataset": scene["dataset"],
                "method": method,
                "role": role,
                "image_count": len(image_paths),
                "image_list": str(image_list.resolve()),
                "output_dir": str(output_dir.resolve()),
            }
            jobs.append(job)
            records.append(
                {
                    "scene_id": scene_id,
                    "scene_key": scene["scene_key"],
                    "dataset": scene["dataset"],
                    "method": method,
                    "role": role,
                    "image_count": len(image_paths),
                    "cache_dir": str(output_dir.resolve()),
                    "token_path": str((output_dir / "camera_and_register_tokens.pt").resolve()),
                }
            )
    return jobs, records


def cache_ready(output_dir: Path) -> bool:
    return all(
        (output_dir / name).is_file()
        for name in (
            "manifest.json",
            "camera_and_register_tokens.pt",
            "register_tokens.pt",
            "register_mean_embedding.json",
            "depth.pt",
            "depth_conf.pt",
            "pose_enc.pt",
        )
    )


def run_cache_jobs_parallel(
    *,
    jobs: list[dict[str, Any]],
    run_dir: Path,
    checkpoint: str,
    image_resolution: int,
    mode: str,
    devices: list[str],
    force: bool,
) -> None:
    if not devices:
        raise ValueError("At least one cache device is required.")
    integration = VGGTOmegaIntegration.discover()
    shards = [[] for _ in devices]
    for index, job in enumerate(jobs):
        shards[index % len(shards)].append(job)

    processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
    for device, shard in zip(devices, shards, strict=True):
        if not shard:
            continue
        jobs_path = run_dir / f"cache_jobs_{device.replace(':', '')}.json"
        jobs_path.write_text(json.dumps(shard, indent=2) + "\n", encoding="utf-8")
        log_path = run_dir / f"cache_{device.replace(':', '')}.log"
        command = [
            str(integration.python),
            "-m",
            "vggt_omega_selector.tools.vggt_batch_cache_runner",
            "--vggt-root",
            str(integration.root),
            "--checkpoint",
            str(integration.checkpoint_path(checkpoint)),
            "--jobs-json",
            str(jobs_path),
            "--image-resolution",
            str(image_resolution),
            "--mode",
            mode,
            "--device",
            device,
            "--include-depth",
            "--include-pose",
        ]
        if force:
            command.append("--force")
        print(json.dumps({"event": "cache_start", "device": device, "jobs": len(shard), "log": str(log_path)}), flush=True)
        log_file = log_path.open("w", encoding="utf-8")
        processes.append(
            (
                device,
                subprocess.Popen(
                    command,
                    cwd=integration.project_root,
                    env=integration.subprocess_env(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
                log_file,
            )
        )

    failed = []
    for device, process, log_file in processes:
        returncode = process.wait()
        log_file.close()
        print(json.dumps({"event": "cache_done", "device": device, "returncode": returncode}), flush=True)
        if returncode != 0:
            failed.append((device, returncode))
    if failed:
        raise SystemExit(f"Cache failed: {failed}")


def validate_cache_records(records: list[dict[str, Any]]) -> None:
    missing = []
    for record in records:
        cache_dir = Path(record["cache_dir"])
        if not cache_ready(cache_dir):
            missing.append(str(cache_dir))
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"Missing hard-label caches ({len(missing)}):\n{preview}")


def compute_native_metrics(
    *,
    records: list[dict[str, Any]],
    max_pixels_per_image: int,
    max_pointmap_points: int,
    epsilon: float,
) -> list[dict[str, Any]]:
    import torch
    from vggt_omega.utils.pose_enc import encoding_to_camera

    native = load_native_module()
    references: dict[str, dict[str, Any]] = {}
    subsets_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["role"] == "reference":
            references[record["scene_id"]] = record
        else:
            subsets_by_scene[record["scene_id"]].append(record)

    rows = []
    for scene_id in sorted(subsets_by_scene):
        reference = references.get(scene_id)
        if reference is None:
            raise RuntimeError(f"{scene_id}: missing full reference")
        ref_cache = native.load_native_cache(Path(reference["cache_dir"]), torch)
        ref_image_index = {native.normalize_image_key(path): index for index, path in enumerate(ref_cache["images"])}
        for subset in sorted(subsets_by_scene[scene_id], key=lambda item: item["method"]):
            sub_cache = native.load_native_cache(Path(subset["cache_dir"]), torch)
            pairs = native.pair_indices(ref_image_index, sub_cache["images"])
            if not pairs:
                raise RuntimeError(f"{scene_id}/{subset['method']}: no common images with full reference")

            ref_indices = [left for left, _right in pairs]
            sub_indices = [right for _left, right in pairs]
            ref_depth = ref_cache["depth"][0, ref_indices, ..., 0].float()
            sub_depth = sub_cache["depth"][0, sub_indices, ..., 0].float()
            ref_conf = ref_cache["depth_conf"][0, ref_indices].float()
            sub_conf = sub_cache["depth_conf"][0, sub_indices].float()
            ref_pose = ref_cache["pose_enc"][0, ref_indices].float()
            sub_pose = sub_cache["pose_enc"][0, sub_indices].float()

            depth_metrics, sampled_pixels = native.compare_depth(
                scene_id=scene_id,
                method=subset["method"],
                ref_depth=ref_depth,
                sub_depth=sub_depth,
                ref_conf=ref_conf,
                sub_conf=sub_conf,
                max_pixels_per_image=max_pixels_per_image,
                epsilon=epsilon,
                torch=torch,
            )
            image_size_hw = tuple(int(value) for value in ref_depth.shape[-2:])
            ref_extrinsics, ref_intrinsics = encoding_to_camera(ref_pose[None], image_size_hw)
            sub_extrinsics, sub_intrinsics = encoding_to_camera(sub_pose[None], image_size_hw)
            ref_extrinsics = ref_extrinsics[0].float()
            sub_extrinsics = sub_extrinsics[0].float()
            ref_intrinsics = ref_intrinsics[0].float()
            sub_intrinsics = sub_intrinsics[0].float()
            pose_metrics, alignment = native.compare_pose(ref_extrinsics, sub_extrinsics, ref_pose, sub_pose)
            point_metrics = native.compare_pointmaps(
                scene_id=scene_id,
                method=subset["method"],
                ref_depth=ref_depth,
                sub_depth=sub_depth,
                ref_extrinsics=ref_extrinsics,
                sub_extrinsics=sub_extrinsics,
                ref_intrinsics=ref_intrinsics,
                sub_intrinsics=sub_intrinsics,
                depth_scales=depth_metrics["per_image_depth_scale"],
                max_pixels_per_image=max_pixels_per_image,
                max_pointmap_points=max_pointmap_points,
                epsilon=epsilon,
                torch=torch,
            )
            similarity = load_register_mean_cosine(Path(reference["cache_dir"]), Path(subset["cache_dir"]))
            rows.append(
                {
                    "scene_id": scene_id,
                    "scene_key": subset["scene_key"],
                    "dataset": subset["dataset"],
                    "method": subset["method"],
                    "image_count": int(subset["image_count"]),
                    "common_images": len(pairs),
                    "sampled_depth_pixels": sampled_pixels,
                    "sampled_pointmap_points": int(point_metrics.pop("sampled_pointmap_points")),
                    "mean_pool_register_cosine": round(float(similarity), 6),
                    **native.round_metric_dict(native.drop_internal(depth_metrics)),
                    **native.round_metric_dict(pose_metrics),
                    **native.round_metric_dict(point_metrics),
                    "alignment_scale": round(float(alignment["scale"]), 8),
                }
            )
        print(json.dumps({"event": "labels_scene_done", "scene_id": scene_id, "subsets": len(subsets_by_scene[scene_id])}), flush=True)
    return rows


def load_native_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "run_stage1_vggt_native_consistency.py"
    spec = importlib.util.spec_from_file_location("stage1_native_consistency_for_hardlabel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_register_mean_cosine(reference_cache: Path, subset_cache: Path) -> float:
    ref = json.loads((reference_cache / "register_mean_embedding.json").read_text(encoding="utf-8"))
    sub = json.loads((subset_cache / "register_mean_embedding.json").read_text(encoding="utf-8"))
    ref_values = [float(value) for value in ref["embedding"]]
    sub_values = [float(value) for value in sub["embedding"]]
    dot = sum(a * b for a, b in zip(ref_values, sub_values, strict=True))
    ref_norm = math.sqrt(sum(a * a for a in ref_values))
    sub_norm = math.sqrt(sum(b * b for b in sub_values))
    return dot / max(ref_norm * sub_norm, 1e-12)


def build_training_labels(metric_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache_lookup = {(record["scene_id"], record["method"]): record for record in records}
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        rows_by_scene[row["scene_id"]].append(row)

    label_rows: list[dict[str, Any]] = []
    for scene_id, rows in sorted(rows_by_scene.items()):
        zscores = {metric: scene_zscores(rows, metric) for metric in PRIMARY_LABEL_METRICS}
        for index, row in enumerate(rows):
            target_error = sum(zscores[metric][index] for metric in PRIMARY_LABEL_METRICS)
            full_record = cache_lookup[(scene_id, "full")]
            subset_record = cache_lookup[(scene_id, row["method"])]
            label_rows.append(
                {
                    "scene_id": scene_id,
                    "scene_key": row["scene_key"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "image_count": row["image_count"],
                    "target_error": round(float(target_error), 8),
                    "full_cache_dir": full_record["cache_dir"],
                    "subset_cache_dir": subset_record["cache_dir"],
                    "full_token_path": full_record["token_path"],
                    "subset_token_path": subset_record["token_path"],
                    **{metric: row[metric] for metric in PRIMARY_LABEL_METRICS},
                    "mean_pool_register_cosine": row["mean_pool_register_cosine"],
                }
            )
    return label_rows


def scene_zscores(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = [float(row[metric]) for row in rows]
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / max(len(values) - 1, 1)
    std = math.sqrt(variance)
    if std <= 1e-12:
        return [0.0 for _value in values]
    return [(value - mean_value) / std for value in values]


def write_label_summary(path: Path, metric_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, int] = {}
    target_values = []
    for row in label_rows:
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        target_values.append(float(row["target_error"]))
    payload = {
        "native_metric_rows": len(metric_rows),
        "label_rows": len(label_rows),
        "labels_by_dataset": by_dataset,
        "primary_label_metrics": list(PRIMARY_LABEL_METRICS),
        "target_error_min": min(target_values) if target_values else None,
        "target_error_max": max(target_values) if target_values else None,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
