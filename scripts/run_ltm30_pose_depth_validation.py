#!/usr/bin/env python3
"""Run VGGT-Omega validation on the LTM30 pose/depth manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_stage1_vggt_native_consistency as native  # noqa: E402
from vggt_omega_selector.backbones.vggt_omega import VGGTOmegaIntegration  # noqa: E402


NATIVE_ERROR_PREFIXES = ("depth_", "pose_", "pointmap_")
GT_ERROR_PREFIXES = ("gt_depth_", "gt_pose_")
LOWER_IS_BETTER_EXCLUDE = {
    "depth_scale_median",
    "depth_scale_mad",
    "depth_conf_mean",
    "gt_depth_scale_median",
    "gt_depth_scale_mad",
    "gt_depth_valid_ratio",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LTM30 pose/depth validation with VGGT-Omega.")
    parser.add_argument(
        "--manifest",
        default="docs/experiments/0002_ltm30_pose_depth_validation/manifest.json",
    )
    parser.add_argument(
        "--cache-root",
        default="caches/vggt_omega/0002_ltm30_pose_depth_validation/native_geometry_images512",
    )
    parser.add_argument(
        "--analysis-dir",
        default="runs/0002_ltm30_pose_depth_validation/native_geometry_images512",
    )
    parser.add_argument(
        "--doc-output-dir",
        default="docs/experiments/0002_ltm30_pose_depth_validation/native_geometry",
    )
    parser.add_argument("--checkpoint", default="512")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-cache-jobs", type=int, default=None)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-pixels-per-image", type=int, default=4096)
    parser.add_argument("--max-pointmap-points", type=int, default=60000)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args(argv)

    project_root = PROJECT_ROOT
    manifest_path = project_root / args.manifest
    cache_root = project_root / args.cache_root
    analysis_dir = project_root / args.analysis_dir
    doc_output_dir = project_root / args.doc_output_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_jobs, records, frame_lookup = build_cache_jobs(
        manifest=manifest,
        cache_root=cache_root,
        analysis_dir=analysis_dir,
        random_seeds=args.random_seeds,
        scene_filter=set(args.scene),
        max_scenes=args.max_scenes,
    )
    if not cache_jobs:
        raise SystemExit("No cache jobs matched.")

    cache_jobs_path = analysis_dir / "cache_jobs.json"
    cache_jobs_path.write_text(json.dumps(cache_jobs, indent=2) + "\n", encoding="utf-8")

    jobs_for_run = cache_jobs[: args.max_cache_jobs] if args.max_cache_jobs else cache_jobs
    if not args.skip_cache:
        run_vggt_batch_cache(
            jobs_json=write_limited_jobs(analysis_dir, jobs_for_run),
            checkpoint=args.checkpoint,
            image_resolution=args.image_resolution,
            mode=args.mode,
            device=args.device,
            force=args.force_cache,
        )
    if args.cache_only:
        print(json.dumps({"cache_jobs": len(jobs_for_run), "all_cache_jobs": len(cache_jobs)}))
        return 0

    validate_cache_outputs(records)
    outputs = analyze(
        records=records,
        frame_lookup=frame_lookup,
        max_pixels_per_image=args.max_pixels_per_image,
        max_pointmap_points=args.max_pointmap_points,
        epsilon=args.epsilon,
    )
    summary = {
        "scene_count": len({record["scene_id"] for record in records}),
        "cache_jobs": len(cache_jobs),
        "records": len(records),
        "register_rows": len(outputs["register_rows"]),
        "native_subset_rows": len(outputs["native_rows"]),
        "gt_rows": len(outputs["gt_rows"]),
        "native_scene_correlation_rows": len(outputs["native_scene_correlations"]),
        "gt_scene_correlation_rows": len(outputs["gt_scene_correlations"]),
        "random_seeds": args.random_seeds,
        "image_resolution": args.image_resolution,
        "mode": args.mode,
        "reference": "LTM30 full split for subset-vs-full metrics; WildRGBD sensor depth/pose for gt_* metrics.",
        "notes": [
            "random20_seed000 reuses manifest random20; later random seeds are deterministic scene-keyed samples from full.",
            "VGGT-native subset metrics compare the same images in subset cache against the full cache.",
            "gt_depth_* aligns VGGT depth to sensor depth with per-image median scale before errors are computed.",
            "gt_pose_* aligns VGGT camera centers to dataset camera_pose centers with one Umeyama similarity per method.",
            "Only text manifests and CSV summaries are written to docs; tensor caches stay under caches/.",
        ],
    }

    for output_dir in (analysis_dir, doc_output_dir):
        write_csv(output_dir / "ltm30_register_similarity.csv", outputs["register_rows"])
        write_csv(output_dir / "ltm30_subset_native_consistency.csv", outputs["native_rows"])
        write_csv(output_dir / "ltm30_gt_geometry_metrics.csv", outputs["gt_rows"])
        write_csv(output_dir / "ltm30_native_scene_correlations.csv", outputs["native_scene_correlations"])
        write_csv(output_dir / "ltm30_gt_scene_correlations.csv", outputs["gt_scene_correlations"])
        write_csv(output_dir / "ltm30_native_correlation_summary.csv", outputs["native_correlation_summary"])
        write_csv(output_dir / "ltm30_gt_correlation_summary.csv", outputs["gt_correlation_summary"])
        write_csv(output_dir / "ltm30_native_method_summary.csv", outputs["native_method_summary"])
        write_csv(output_dir / "ltm30_gt_method_summary.csv", outputs["gt_method_summary"])
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_summary_md(output_dir / "summary.md", summary, outputs)

    print(json.dumps(summary, indent=2))
    return 0


def safe_scene_id(scene_key: str) -> str:
    return scene_key.replace("/", "__")


def stable_seed(base_seed: int, scene_key: str, random_index: int) -> int:
    payload = f"{base_seed}:{random_index}:{scene_key}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def sample_random_ids(scene: dict[str, Any], random_seed: int, random_index: int) -> list[str]:
    if random_index == 0 and "random20" in scene["splits"]:
        return list(scene["splits"]["random20"])
    full_ids = list(scene["splits"]["full"])
    count = max(1, int(round(len(full_ids) * 0.20)))
    rng = np.random.default_rng(stable_seed(random_seed, scene["scene_key"], random_index))
    indices = sorted(int(i) for i in rng.choice(len(full_ids), size=count, replace=False))
    return [full_ids[i] for i in indices]


def build_cache_jobs(
    *,
    manifest: dict[str, Any],
    cache_root: Path,
    analysis_dir: Path,
    random_seeds: int,
    scene_filter: set[str],
    max_scenes: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    cache_jobs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    frame_lookup: dict[str, dict[str, dict[str, Any]]] = {}
    selected_scenes = []
    for scene in manifest["scenes"]:
        scene_key = scene["scene_key"]
        scene_id = safe_scene_id(scene_key)
        if scene_filter and scene_key not in scene_filter and scene_id not in scene_filter:
            continue
        selected_scenes.append(scene)
        if max_scenes and len(selected_scenes) >= max_scenes:
            break

    base_seed = int(manifest["selection_policy"]["random_seed"])
    for scene in selected_scenes:
        scene_key = scene["scene_key"]
        scene_id = safe_scene_id(scene_key)
        frames_by_id = {frame["frame_id"]: frame for frame in scene["frames"]}
        frame_lookup[scene_id] = {}

        method_to_ids: dict[str, list[str]] = {"full": list(scene["splits"]["full"])}
        for seed_index in range(random_seeds):
            method_to_ids[f"random20_seed{seed_index:03d}"] = sample_random_ids(scene, base_seed, seed_index)
        method_to_ids["uniform20"] = list(scene["splits"]["uniform20"])

        for method, frame_ids in method_to_ids.items():
            role = "reference" if method == "full" else "subset"
            method_frames = [frames_by_id[frame_id] for frame_id in frame_ids]
            image_paths = [str((PROJECT_ROOT / frame["image_path"]).resolve()) for frame in method_frames]
            for frame, image_path in zip(method_frames, image_paths, strict=True):
                frame_lookup[scene_id][str(Path(image_path).resolve())] = frame

            image_list = analysis_dir / "image_lists" / scene_id / f"{method}.txt"
            image_list.parent.mkdir(parents=True, exist_ok=True)
            image_list.write_text("\n".join(image_paths) + "\n", encoding="utf-8")

            output_dir = cache_root / scene_id / method
            job = {
                "id": f"{scene_id}/{method}",
                "scene_id": scene_id,
                "scene_key": scene_key,
                "method": method,
                "role": role,
                "image_count": len(image_paths),
                "image_list": str(image_list.resolve()),
                "output_dir": str(output_dir.resolve()),
            }
            cache_jobs.append(job)
            records.append(
                {
                    "scene_id": scene_id,
                    "scene_key": scene_key,
                    "method": method,
                    "role": role,
                    "image_count": len(image_paths),
                    "image_list": str(image_list.resolve()),
                    "cache_dir": str(output_dir.resolve()),
                }
            )
    return cache_jobs, records, frame_lookup


def write_limited_jobs(analysis_dir: Path, jobs: list[dict[str, Any]]) -> Path:
    path = analysis_dir / "cache_jobs_to_run.json"
    path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return path


def run_vggt_batch_cache(
    *,
    jobs_json: Path,
    checkpoint: str,
    image_resolution: int,
    mode: str,
    device: str,
    force: bool,
) -> None:
    integration = VGGTOmegaIntegration.discover()
    command = [
        str(integration.python),
        "-m",
        "vggt_omega_selector.tools.vggt_batch_cache_runner",
        "--vggt-root",
        str(integration.root),
        "--checkpoint",
        str(integration.checkpoint_path(checkpoint)),
        "--jobs-json",
        str(jobs_json),
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
    completed = subprocess.run(command, cwd=integration.project_root, env=integration.subprocess_env(), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def validate_cache_outputs(records: list[dict[str, Any]]) -> None:
    required = ["manifest.json", "register_mean_embedding.json", "depth.pt", "depth_conf.pt", "pose_enc.pt"]
    missing: list[str] = []
    for record in records:
        cache_dir = Path(record["cache_dir"])
        for filename in required:
            if not (cache_dir / filename).is_file():
                missing.append(f"{record['scene_id']}/{record['method']}:{filename}")
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"Missing cache outputs ({len(missing)} files):\n{preview}")


def analyze(
    *,
    records: list[dict[str, Any]],
    frame_lookup: dict[str, dict[str, dict[str, Any]]],
    max_pixels_per_image: int,
    max_pointmap_points: int,
    epsilon: float,
) -> dict[str, list[dict[str, Any]]]:
    import torch
    from vggt_omega.utils.pose_enc import encoding_to_camera

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["scene_id"]].append(record)

    register_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []

    for scene_id, scene_records in sorted(grouped.items()):
        reference = next(record for record in scene_records if record["role"] == "reference")
        ref_cache = native.load_native_cache(Path(reference["cache_dir"]), torch)
        ref_embedding = load_embedding(Path(reference["cache_dir"]))
        ref_index = {native.normalize_image_key(path): index for index, path in enumerate(ref_cache["images"])}

        for record in sorted(scene_records, key=lambda item: method_order(item["method"])):
            cache_dir = Path(record["cache_dir"])
            embedding = load_embedding(cache_dir)
            register_cosine = cosine(embedding, ref_embedding)
            register_rows.append(
                {
                    "scene_id": scene_id,
                    "scene_key": record["scene_key"],
                    "method": record["method"],
                    "role": record["role"],
                    "image_count": record["image_count"],
                    "register_mean_cosine": round(register_cosine, 8),
                }
            )

            cache = ref_cache if record["role"] == "reference" else native.load_native_cache(cache_dir, torch)
            gt_rows.append(
                analyze_against_gt(
                    record=record,
                    cache=cache,
                    frame_lookup=frame_lookup[scene_id],
                    register_cosine=register_cosine,
                    max_pixels_per_image=max_pixels_per_image,
                    epsilon=epsilon,
                    torch=torch,
                    encoding_to_camera=encoding_to_camera,
                )
            )

            if record["role"] == "reference":
                continue
            sub_cache = cache
            pairs = native.pair_indices(ref_index, sub_cache["images"])
            if not pairs:
                raise RuntimeError(f"{scene_id}/{record['method']}: no common images with full cache")
            ref_indices = [left for left, _right in pairs]
            sub_indices = [right for _left, right in pairs]

            ref_depth = ref_cache["depth"][0, ref_indices, ..., 0].float()
            sub_depth = sub_cache["depth"][0, sub_indices, ..., 0].float()
            ref_conf = ref_cache["depth_conf"][0, ref_indices].float()
            sub_conf = sub_cache["depth_conf"][0, sub_indices].float()
            ref_pose = ref_cache["pose_enc"][0, ref_indices].float()
            sub_pose = sub_cache["pose_enc"][0, sub_indices].float()

            depth_metrics, sampled_depth_pixels = native.compare_depth(
                scene_id=scene_id,
                method=record["method"],
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
                method=record["method"],
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

            native_rows.append(
                {
                    "scene_id": scene_id,
                    "scene_key": record["scene_key"],
                    "method": record["method"],
                    "image_count": record["image_count"],
                    "common_images": len(pairs),
                    "sampled_depth_pixels": sampled_depth_pixels,
                    "sampled_pointmap_points": int(point_metrics.pop("sampled_pointmap_points")),
                    "register_mean_cosine": round(register_cosine, 8),
                    **round_dict(native.drop_internal(depth_metrics)),
                    **round_dict(pose_metrics),
                    **round_dict(point_metrics),
                    "alignment_scale": round(float(alignment["scale"]), 8),
                }
            )

    native_scene_correlations = build_scene_correlations(native_rows, NATIVE_ERROR_PREFIXES)
    gt_scene_correlations = build_scene_correlations(gt_rows, GT_ERROR_PREFIXES)
    return {
        "register_rows": register_rows,
        "native_rows": native_rows,
        "gt_rows": gt_rows,
        "native_scene_correlations": native_scene_correlations,
        "gt_scene_correlations": gt_scene_correlations,
        "native_correlation_summary": summarize_correlations(native_scene_correlations),
        "gt_correlation_summary": summarize_correlations(gt_scene_correlations),
        "native_method_summary": summarize_by_method(native_rows, NATIVE_ERROR_PREFIXES),
        "gt_method_summary": summarize_by_method(gt_rows, GT_ERROR_PREFIXES),
    }


def load_embedding(cache_dir: Path) -> np.ndarray:
    payload = json.loads((cache_dir / "register_mean_embedding.json").read_text(encoding="utf-8"))
    return np.asarray(payload["embedding"], dtype=np.float64)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def analyze_against_gt(
    *,
    record: dict[str, Any],
    cache: dict[str, Any],
    frame_lookup: dict[str, dict[str, Any]],
    register_cosine: float,
    max_pixels_per_image: int,
    epsilon: float,
    torch: Any,
    encoding_to_camera: Any,
) -> dict[str, Any]:
    pred_depth = cache["depth"][0, ..., 0].float()
    pred_pose = cache["pose_enc"][0].float()
    image_size_hw = tuple(int(value) for value in pred_depth.shape[-2:])
    pred_extrinsics, _pred_intrinsics = encoding_to_camera(pred_pose[None], image_size_hw)
    pred_extrinsics = pred_extrinsics[0].float()

    frames = [frame_lookup[native.normalize_image_key(path)] for path in cache["images"]]
    depth_metrics = compare_gt_depth(
        scene_id=record["scene_id"],
        method=record["method"],
        pred_depth=pred_depth,
        frames=frames,
        max_pixels_per_image=max_pixels_per_image,
        epsilon=epsilon,
    )
    pose_metrics = compare_gt_pose(
        pred_pose=pred_pose,
        pred_extrinsics=pred_extrinsics,
        frames=frames,
        image_size_hw=image_size_hw,
        torch=torch,
    )
    return {
        "scene_id": record["scene_id"],
        "scene_key": record["scene_key"],
        "method": record["method"],
        "role": record["role"],
        "image_count": record["image_count"],
        "register_mean_cosine": round(register_cosine, 8),
        **round_dict(depth_metrics),
        **round_dict(pose_metrics),
    }


def compare_gt_depth(
    *,
    scene_id: str,
    method: str,
    pred_depth: Any,
    frames: list[dict[str, Any]],
    max_pixels_per_image: int,
    epsilon: float,
) -> dict[str, float]:
    abs_rel_values: list[np.ndarray] = []
    log_values: list[np.ndarray] = []
    scales: list[float] = []
    valid_ratios: list[float] = []
    pred_h, pred_w = int(pred_depth.shape[-2]), int(pred_depth.shape[-1])

    for frame_index, frame in enumerate(frames):
        gt_depth = load_depth_png_resized(PROJECT_ROOT / frame["depth_path"], pred_w, pred_h)
        pred = pred_depth[frame_index].cpu().numpy().astype(np.float64)
        indices = native.sample_flat_indices(
            count=pred_h * pred_w,
            max_count=max_pixels_per_image,
            seed_key=f"{scene_id}:{method}:gt_depth:{frame_index}",
        )
        ys = indices // pred_w
        xs = indices % pred_w
        gt = gt_depth[ys, xs].astype(np.float64)
        pd = pred[ys, xs]
        mask = np.isfinite(gt) & np.isfinite(pd) & (gt > 0) & (pd > epsilon)
        valid_ratios.append(float(np.mean(mask)))
        if int(mask.sum()) < 8:
            continue
        gt = gt[mask]
        pd = pd[mask]
        scale = float(np.median(gt / np.clip(pd, epsilon, None)))
        aligned = pd * scale
        abs_rel_values.append(np.abs(aligned - gt) / np.clip(np.abs(gt), epsilon, None))
        log_values.append(np.log(np.clip(aligned, epsilon, None)) - np.log(np.clip(gt, epsilon, None)))
        scales.append(scale)

    if not abs_rel_values:
        raise RuntimeError(f"{scene_id}/{method}: no valid gt depth samples")
    abs_rel_all = np.concatenate(abs_rel_values)
    log_all = np.concatenate(log_values)
    scale_array = np.asarray(scales, dtype=np.float64)
    return {
        "gt_depth_absrel_mean": float(np.mean(abs_rel_all)),
        "gt_depth_absrel_median": float(np.median(abs_rel_all)),
        "gt_depth_log_rmse": float(np.sqrt(np.mean(np.square(log_all)))),
        "gt_depth_scale_median": float(np.median(scale_array)),
        "gt_depth_scale_mad": float(np.median(np.abs(scale_array - np.median(scale_array)))),
        "gt_depth_valid_ratio": float(np.mean(valid_ratios)),
    }


def load_depth_png_resized(path: Path, width: int, height: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        image = image.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.float64)


def compare_gt_pose(
    *,
    pred_pose: Any,
    pred_extrinsics: Any,
    frames: list[dict[str, Any]],
    image_size_hw: tuple[int, int],
    torch: Any,
) -> dict[str, float]:
    pred_centers = native.camera_centers(pred_extrinsics).numpy()
    pred_rot_c2w = np.transpose(pred_extrinsics[:, :3, :3].numpy(), (0, 2, 1))

    gt_centers = []
    gt_rot_c2w = []
    gt_fovs = []
    for frame in frames:
        pose_path = PROJECT_ROOT / frame["pose_path"]
        with np.load(pose_path) as data:
            camera_pose = np.asarray(data["camera_pose"], dtype=np.float64)
            camera_intrinsics = np.asarray(data["camera_intrinsics"], dtype=np.float64)
        gt_centers.append(camera_pose[:3, 3])
        gt_rot_c2w.append(camera_pose[:3, :3])
        gt_fovs.append(gt_fov_for_frame(frame, camera_intrinsics, image_size_hw))

    gt_centers_array = np.asarray(gt_centers, dtype=np.float64)
    gt_rot_array = np.asarray(gt_rot_c2w, dtype=np.float64)
    alignment = native.umeyama_alignment(pred_centers, gt_centers_array)
    pred_centers_aligned = native.apply_similarity(pred_centers, alignment)
    scale = native.bbox_diagonal(gt_centers_array)
    center_errors = np.linalg.norm(pred_centers_aligned - gt_centers_array, axis=1) / scale

    align_rot = alignment["rotation"]
    rotation_errors = []
    for pred_rot, gt_rot in zip(pred_rot_c2w, gt_rot_array, strict=True):
        pred_aligned = align_rot @ pred_rot
        delta = pred_aligned @ gt_rot.T
        trace_value = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(math.degrees(math.acos(trace_value)))

    gt_fov = torch.as_tensor(np.asarray(gt_fovs, dtype=np.float32))
    fov_error = np.abs(pred_pose[:, 7:9].cpu().numpy() - gt_fov.numpy())
    return {
        "gt_pose_center_rmse_norm": float(np.sqrt(np.mean(np.square(center_errors)))),
        "gt_pose_center_median_norm": float(np.median(center_errors)),
        "gt_pose_center_p90_norm": float(np.quantile(center_errors, 0.90)),
        "gt_pose_rotation_mean_deg": float(np.mean(rotation_errors)),
        "gt_pose_rotation_median_deg": float(np.median(rotation_errors)),
        "gt_pose_fov_abs_mean": float(np.mean(fov_error)),
        "gt_pose_alignment_scale": float(alignment["scale"]),
    }


def gt_fov_for_frame(frame: dict[str, Any], intrinsics: np.ndarray, image_size_hw: tuple[int, int]) -> tuple[float, float]:
    from PIL import Image

    pred_h, pred_w = image_size_hw
    with Image.open(PROJECT_ROOT / frame["image_path"]) as image:
        original_w, original_h = image.size
    fx = intrinsics[0, 0] * pred_w / max(original_w, 1)
    fy = intrinsics[1, 1] * pred_h / max(original_h, 1)
    fov_h = 2 * math.atan((pred_h / 2.0) / max(float(fy), 1e-12))
    fov_w = 2 * math.atan((pred_w / 2.0) / max(float(fx), 1e-12))
    return fov_h, fov_w


def build_scene_correlations(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metric_names = [
        name
        for name in rows[0]
        if name.startswith(prefixes) and name not in LOWER_IS_BETTER_EXCLUDE and not name.endswith("_alignment_scale")
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)

    scene_rows = []
    for scene_id, items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        for metric in metric_names:
            cosine_values = [float(item["register_mean_cosine"]) for item in items]
            metric_values = [float(item[metric]) for item in items]
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "scene_key": items[0]["scene_key"],
                    "metric": metric,
                    "expected_direction": "negative",
                    "n": len(items),
                    "spearman": round(native.spearman(cosine_values, metric_values), 6),
                    "pearson": round(native.pearson(cosine_values, metric_values), 6),
                    "best_cosine_method": max(items, key=lambda item: float(item["register_mean_cosine"]))["method"],
                    "best_metric_method": min(items, key=lambda item: float(item[metric]))["method"],
                }
            )
    return scene_rows


def summarize_correlations(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        grouped[row["metric"]].append(row)
    rows = []
    for metric, items in sorted(grouped.items()):
        spearman_values = [float(item["spearman"]) for item in items]
        pearson_values = [float(item["pearson"]) for item in items]
        rows.append(
            {
                "metric": metric,
                "expected_direction": "negative",
                "scenes": len(items),
                "mean_spearman": round(float(np.mean(spearman_values)), 6),
                "spearman_expected_sign": f"{sum(value < 0 for value in spearman_values)}/{len(items)}",
                "mean_pearson": round(float(np.mean(pearson_values)), 6),
                "pearson_expected_sign": f"{sum(value < 0 for value in pearson_values)}/{len(items)}",
                "best_match": f"{sum(item['best_cosine_method'] == item['best_metric_method'] for item in items)}/{len(items)}",
            }
        )
    return rows


def summarize_by_method(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metric_names = [
        name
        for name in rows[0]
        if name.startswith(prefixes) and name not in LOWER_IS_BETTER_EXCLUDE and not name.endswith("_alignment_scale")
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary = []
    for method, items in sorted(grouped.items(), key=lambda item: method_order(item[0])):
        output: dict[str, Any] = {
            "method": method,
            "runs": len(items),
            "register_mean_cosine_mean": round(float(np.mean([float(item["register_mean_cosine"]) for item in items])), 8),
        }
        for metric in metric_names:
            output[f"{metric}_mean"] = round(float(np.mean([float(item[metric]) for item in items])), 8)
        summary.append(output)
    return summary


def method_order(method: str) -> int:
    if method == "full":
        return 0
    if method.startswith("random20_seed"):
        try:
            return 10 + int(method.rsplit("seed", 1)[1])
        except ValueError:
            return 19
    if method == "uniform20":
        return 100
    return 999


def round_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: round(float(value), 8) for key, value in metrics.items()}


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


def write_summary_md(path: Path, summary: dict[str, Any], outputs: dict[str, list[dict[str, Any]]]) -> None:
    gt_top = outputs["gt_correlation_summary"][:]
    native_top = outputs["native_correlation_summary"][:]
    lines = [
        "# LTM30 Native Geometry Validation",
        "",
        f"- Scenes: {summary['scene_count']}",
        f"- Cache jobs: {summary['cache_jobs']}",
        f"- Random seeds: {summary['random_seeds']}",
        f"- GT rows: {summary['gt_rows']}",
        f"- Native subset rows: {summary['native_subset_rows']}",
        f"- Image resolution: {summary['image_resolution']} ({summary['mode']})",
        "",
        "## GT Correlation Summary",
        "",
        "| Metric | Scenes | Mean Spearman | Sign | Best Match |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in gt_top:
        lines.append(
            f"| `{row['metric']}` | {row['scenes']} | {row['mean_spearman']} | "
            f"{row['spearman_expected_sign']} | {row['best_match']} |"
        )
    lines.extend(
        [
            "",
            "## Native Subset Correlation Summary",
            "",
            "| Metric | Scenes | Mean Spearman | Sign | Best Match |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in native_top:
        lines.append(
            f"| `{row['metric']}` | {row['scenes']} | {row['mean_spearman']} | "
            f"{row['spearman_expected_sign']} | {row['best_match']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
