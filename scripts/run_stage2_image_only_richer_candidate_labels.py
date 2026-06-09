#!/usr/bin/env python3
"""Build richer 0005 candidate labels without using VGGT features as student input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vggt_omega_selector.backbones.vggt_omega import VGGTOmegaIntegration  # noqa: E402


PRIMARY_LABEL_METRICS = ("pose_rotation_mean_deg", "pointmap_rmse_norm", "depth_log_rmse")


@dataclass(frozen=True)
class SceneSource:
    scene_id: str
    scene_key: str
    dataset: str
    full_image_list: Path
    full_images: list[Path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate richer hard-native labels for 0005 image-only selector.")
    parser.add_argument(
        "--base-labels-csv",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv",
    )
    parser.add_argument(
        "--base-metrics-csv",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_native_metrics.csv",
    )
    parser.add_argument(
        "--base-cache-jobs-json",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json",
    )
    parser.add_argument("--convnext-feature-cache", default="caches/image_features/0005/hardlabel300_convnext_tiny")
    parser.add_argument("--dinov2-feature-cache", default="caches/image_features/0005/hardlabel300_dinov2_vits14")
    parser.add_argument("--run-dir", default="runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300")
    parser.add_argument(
        "--cache-root",
        default="caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_images512",
    )
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--checkpoint", default="512")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--cache-devices", default="cuda:0,cuda:1")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-pixels-per-image", type=int, default=1024)
    parser.add_argument("--max-pointmap-points", type=int, default=60000)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args(argv)

    run_dir = resolve(args.run_dir)
    cache_root = resolve(args.cache_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    scenes = load_scene_sources(
        base_labels_csv=resolve(args.base_labels_csv),
        base_cache_jobs_json=resolve(args.base_cache_jobs_json),
        candidate_tag=args.candidate_tag,
        limit_scenes=args.limit_scenes,
        seed=args.seed,
    )
    jobs, records = build_richer_candidate_jobs(
        scenes=scenes,
        run_dir=run_dir,
        cache_root=cache_root,
        convnext_feature_cache=resolve(args.convnext_feature_cache),
        dinov2_feature_cache=resolve(args.dinov2_feature_cache),
        candidate_tag=args.candidate_tag,
        seed=args.seed,
    )
    (run_dir / "richer_cache_jobs.json").write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "richer_cache_records.json").write_text(
        json.dumps({"records": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_candidate_summary(run_dir / "richer_candidate_summary.json", scenes, records)

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
            print(json.dumps({"event": "cache_skip", "reason": "all richer jobs already cached", "jobs": len(jobs)}), flush=True)

    if args.cache_only:
        print(json.dumps({"event": "cache_only_done", "jobs": len(jobs), "records": len(records)}), flush=True)
        return 0

    validate_cache_records(records)

    richer_metrics = compute_native_metrics(
        records=records,
        max_pixels_per_image=args.max_pixels_per_image,
        max_pointmap_points=args.max_pointmap_points,
        epsilon=args.epsilon,
    )
    write_csv(run_dir / "richer_hardlabel_native_metrics.csv", richer_metrics)

    base_metric_rows = read_csv(resolve(args.base_metrics_csv))
    merged_metric_rows = merge_metric_rows(base_metric_rows, richer_metrics, candidate_tag=args.candidate_tag)
    write_csv(run_dir / "merged_hardlabel_native_metrics.csv", merged_metric_rows)

    base_jobs = json.loads(resolve(args.base_cache_jobs_json).read_text(encoding="utf-8"))
    merged_jobs = merge_jobs(base_jobs, jobs, candidate_tag=args.candidate_tag)
    (run_dir / "merged_cache_jobs.json").write_text(json.dumps(merged_jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    merged_records = merge_records(records_from_jobs(base_jobs), records, candidate_tag=args.candidate_tag)
    merged_labels = build_training_labels(merged_metric_rows, merged_records)
    write_csv(run_dir / "merged_hardlabel_train_labels.csv", merged_labels)
    write_label_summary(run_dir / "merged_hardlabel_summary.json", merged_metric_rows, merged_labels)

    print(
        json.dumps(
            {
                "event": "richer_labels_done",
                "scenes": len(scenes),
                "richer_metric_rows": len(richer_metrics),
                "merged_metric_rows": len(merged_metric_rows),
                "merged_label_rows": len(merged_labels),
                "labels_csv": str((run_dir / "merged_hardlabel_train_labels.csv").resolve()),
                "cache_jobs_json": str((run_dir / "merged_cache_jobs.json").resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_scene_sources(
    *,
    base_labels_csv: Path,
    base_cache_jobs_json: Path,
    candidate_tag: str,
    limit_scenes: int | None,
    seed: int,
) -> list[SceneSource]:
    wanted = set()
    methods_by_scene: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(base_labels_csv):
        method = row["method"]
        if is_base_candidate_method(method, candidate_tag):
            wanted.add(row["scene_id"])
            methods_by_scene[row["scene_id"]].add(method)

    jobs = json.loads(base_cache_jobs_json.read_text(encoding="utf-8"))
    scenes = []
    for job in jobs:
        if job.get("method") != "full":
            continue
        scene_id = job["scene_id"]
        if scene_id not in wanted or f"uniform{candidate_tag}" not in methods_by_scene[scene_id]:
            continue
        image_list = resolve(job["image_list"])
        scenes.append(
            SceneSource(
                scene_id=scene_id,
                scene_key=job["scene_key"],
                dataset=job["dataset"],
                full_image_list=image_list,
                full_images=read_image_list(image_list),
            )
        )
    scenes = sorted(scenes, key=lambda scene: (scene.dataset, scene.scene_id))
    rng = random.Random(seed)
    rng.shuffle(scenes)
    if limit_scenes is not None:
        scenes = scenes[:limit_scenes]
    return scenes


def build_richer_candidate_jobs(
    *,
    scenes: list[SceneSource],
    run_dir: Path,
    cache_root: Path,
    convnext_feature_cache: Path,
    dinov2_feature_cache: Path,
    candidate_tag: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = []
    records = []
    for scene in scenes:
        candidates = build_scene_candidates(
            scene=scene,
            convnext_feature_cache=convnext_feature_cache,
            dinov2_feature_cache=dinov2_feature_cache,
            candidate_tag=candidate_tag,
            seed=seed,
        )
        all_methods = {"full": list(range(len(scene.full_images))), **candidates}
        for method, indices in all_methods.items():
            image_paths = [scene.full_images[index] for index in indices]
            image_list = run_dir / "image_lists" / scene.scene_id / f"{method}.txt"
            image_list.parent.mkdir(parents=True, exist_ok=True)
            image_list.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")
            output_dir = cache_root / scene.scene_id / method
            role = "reference" if method == "full" else "subset"
            job = {
                "id": f"{scene.scene_id}/{method}",
                "scene_id": scene.scene_id,
                "scene_key": scene.scene_key,
                "dataset": scene.dataset,
                "method": method,
                "role": role,
                "image_count": len(image_paths),
                "image_list": str(image_list.resolve()),
                "output_dir": str(output_dir.resolve()),
            }
            jobs.append(job)
            records.append(
                {
                    "scene_id": scene.scene_id,
                    "scene_key": scene.scene_key,
                    "dataset": scene.dataset,
                    "method": method,
                    "role": role,
                    "image_count": len(image_paths),
                    "cache_dir": str(output_dir.resolve()),
                    "token_path": str((output_dir / "camera_and_register_tokens.pt").resolve()),
                }
            )
    return jobs, records


def build_scene_candidates(
    *,
    scene: SceneSource,
    convnext_feature_cache: Path,
    dinov2_feature_cache: Path,
    candidate_tag: str,
    seed: int,
) -> dict[str, list[int]]:
    count = len(scene.full_images)
    ratio = int(candidate_tag) / 100.0
    k = max(1, int(round(count * ratio)))
    candidates: dict[str, list[int]] = {}

    for jitter_seed in range(5):
        name = f"uniform_jitter{candidate_tag}_seed{jitter_seed:03d}"
        candidates[name] = uniform_jitter_indices(
            count,
            k,
            stable_seed(seed, scene.scene_id, name),
        )

    conv_features = load_feature_matrix(convnext_feature_cache / f"{scene.scene_id}.pt")
    dino_features = load_feature_matrix(dinov2_feature_cache / f"{scene.scene_id}.pt")
    candidates[f"convnext_kcenter{candidate_tag}_seed000"] = kcenter_indices(
        conv_features,
        k,
        stable_seed(seed, scene.scene_id, f"convnext_kcenter{candidate_tag}"),
    )
    candidates[f"dinov2_kcenter{candidate_tag}_seed000"] = kcenter_indices(
        dino_features,
        k,
        stable_seed(seed, scene.scene_id, f"dinov2_kcenter{candidate_tag}"),
    )
    candidates[f"motion_spread{candidate_tag}_seed000"] = motion_spread_indices(
        dino_features,
        k,
        stable_seed(seed, scene.scene_id, f"motion_spread{candidate_tag}"),
    )
    return candidates


def uniform_indices(count: int, sample_count: int) -> list[int]:
    if sample_count >= count:
        return list(range(count))
    if sample_count <= 1:
        return [0]
    indices: list[int] = []
    seen: set[int] = set()
    for i in range(sample_count):
        idx = round(i * (count - 1) / (sample_count - 1))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    fill = 0
    while len(indices) < sample_count:
        if fill not in seen:
            indices.append(fill)
            seen.add(fill)
        fill += 1
    return sorted(indices)


def uniform_jitter_indices(count: int, k: int, seed: int) -> list[int]:
    base = uniform_indices(count, k)
    if k >= count:
        return base
    rng = random.Random(seed)
    stride = max(1, int(round(count / max(k, 1))))
    radius = max(1, stride // 3)
    selected = []
    seen = set()
    for idx in base:
        lo = max(0, idx - radius)
        hi = min(count - 1, idx + radius)
        jittered = rng.randint(lo, hi)
        if jittered not in seen:
            selected.append(jittered)
            seen.add(jittered)
    fill_order = base + list(range(count))
    for idx in fill_order:
        if len(selected) >= k:
            break
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    return sorted(selected[:k])


def kcenter_indices(features: torch.Tensor, k: int, seed: int) -> list[int]:
    count = int(features.shape[0])
    if k >= count:
        return list(range(count))
    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    selected = [0, count - 1] if k > 1 else [0]
    rng = random.Random(seed)
    if k > 2:
        selected.append(rng.randrange(count))
    selected = unique_keep_order(selected)
    min_dist = torch.full((count,), float("inf"))
    while len(selected) < k:
        current = normalized[selected]
        dist = 1.0 - normalized @ current.T
        min_dist = torch.minimum(min_dist, dist.min(dim=1).values)
        min_dist[selected] = -1.0
        selected.append(int(torch.argmax(min_dist).item()))
    return sorted(selected[:k])


def motion_spread_indices(features: torch.Tensor, k: int, seed: int) -> list[int]:
    count = int(features.shape[0])
    if k >= count:
        return list(range(count))
    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    deltas = 1.0 - (normalized[1:] * normalized[:-1]).sum(dim=-1)
    ranked_edges = torch.argsort(deltas, descending=True).tolist()
    selected = {0, count - 1}
    for edge in ranked_edges:
        if len(selected) >= max(2, k // 2):
            break
        selected.add(edge)
        selected.add(min(edge + 1, count - 1))
    rng = random.Random(seed)
    jittered_uniform = uniform_jitter_indices(count, k, rng.randrange(2**31))
    for idx in jittered_uniform:
        selected.add(idx)
        if len(selected) >= k:
            break
    if len(selected) < k:
        for idx in range(count):
            selected.add(idx)
            if len(selected) >= k:
                break
    return sorted(selected)[:k]


def load_feature_matrix(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    features = payload["frame_features"].float()
    stats = payload.get("image_stats")
    if stats is not None:
        features = torch.cat([features, stats.float()], dim=-1)
    return features


def stable_seed(seed: int, scene_id: str, salt: str) -> int:
    payload = f"{seed}:{scene_id}:{salt}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def unique_keep_order(values: list[int]) -> list[int]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


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
        jobs_path.write_text(json.dumps(shard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
        raise RuntimeError(f"Missing richer hard-label caches ({len(missing)}):\n" + "\n".join(missing[:20]))


def compute_native_metrics(
    *,
    records: list[dict[str, Any]],
    max_pixels_per_image: int,
    max_pointmap_points: int,
    epsilon: float,
) -> list[dict[str, Any]]:
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
    spec = importlib.util.spec_from_file_location("stage1_native_consistency_for_0005", path)
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


def merge_metric_rows(
    base_rows: list[dict[str, Any]],
    richer_rows: list[dict[str, Any]],
    candidate_tag: str,
) -> list[dict[str, Any]]:
    out = [row for row in base_rows if is_base_candidate_method(row["method"], candidate_tag)]
    out.extend(richer_rows)
    return sorted(out, key=lambda row: (row["scene_id"], row["method"]))


def merge_jobs(base_jobs: list[dict[str, Any]], richer_jobs: list[dict[str, Any]], candidate_tag: str) -> list[dict[str, Any]]:
    out = []
    for job in base_jobs:
        method = job["method"]
        if method == "full" or is_base_candidate_method(method, candidate_tag):
            out.append(job)
    out.extend(richer_jobs)
    return sorted(out, key=lambda job: (job["scene_id"], job["method"], job["output_dir"]))


def merge_records(base_records: list[dict[str, Any]], richer_records: list[dict[str, Any]], candidate_tag: str) -> list[dict[str, Any]]:
    out = []
    for record in base_records:
        method = record["method"]
        if method == "full" or is_base_candidate_method(method, candidate_tag):
            out.append(record)
    out.extend(richer_records)
    return sorted(out, key=lambda record: (record["scene_id"], record["method"], record["cache_dir"]))


def records_from_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for job in jobs:
        output_dir = Path(job["output_dir"])
        records.append(
            {
                "scene_id": job["scene_id"],
                "scene_key": job["scene_key"],
                "dataset": job["dataset"],
                "method": job["method"],
                "role": job["role"],
                "image_count": int(job["image_count"]),
                "cache_dir": str(output_dir.resolve()),
                "token_path": str((output_dir / "camera_and_register_tokens.pt").resolve()),
            }
        )
    return records


def is_base_candidate_method(method: str, tag: str) -> bool:
    return method == f"uniform{tag}" or method.startswith(f"random{tag}_") or method.startswith(f"contiguous{tag}_")


def read_image_list(path: Path) -> list[Path]:
    return [resolve(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_summary(path: Path, scenes: list[SceneSource], records: list[dict[str, Any]]) -> None:
    by_method: dict[str, int] = defaultdict(int)
    by_dataset: dict[str, int] = defaultdict(int)
    for scene in scenes:
        by_dataset[scene.dataset] += 1
    for record in records:
        if record["method"] != "full":
            by_method[record["method"]] += 1
    payload = {
        "scenes": len(scenes),
        "by_dataset": dict(sorted(by_dataset.items())),
        "subset_jobs": sum(by_method.values()),
        "full_jobs": len(scenes),
        "by_method": dict(sorted(by_method.items())),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_label_summary(path: Path, metric_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, int] = defaultdict(int)
    oracle_counts: dict[str, int] = defaultdict(int)
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_values = []
    for row in label_rows:
        by_dataset[row["dataset"]] += 1
        rows_by_scene[row["scene_id"]].append(row)
        target_values.append(float(row["target_error"]))
    for rows in rows_by_scene.values():
        best = min(rows, key=lambda row: float(row["target_error"]))
        method = best["method"]
        if method.startswith("random20_"):
            method = "random20"
        elif method.startswith("uniform_jitter20_"):
            method = "uniform_jitter20"
        elif method.startswith("contiguous20_"):
            method = "contiguous20"
        oracle_counts[method] += 1
    payload = {
        "native_metric_rows": len(metric_rows),
        "label_rows": len(label_rows),
        "labels_by_dataset": dict(sorted(by_dataset.items())),
        "primary_label_metrics": list(PRIMARY_LABEL_METRICS),
        "target_error_min": min(target_values) if target_values else None,
        "target_error_max": max(target_values) if target_values else None,
        "oracle_counts": dict(sorted(oracle_counts.items())),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
