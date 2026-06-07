#!/usr/bin/env python3
"""Compare VGGT-native depth, point-map, and pose predictions across subsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vggt_omega_selector.backbones.vggt_omega import VGGTOmegaIntegration


METHODS = [f"random_ratio_seed{seed:03d}" for seed in range(5)] + ["uniform_stride_ratio"]
ERROR_METRIC_PREFIXES = ("depth_", "pose_", "pointmap_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 1 VGGT-native consistency analysis.")
    parser.add_argument(
        "--source-cache-jobs",
        default="runs/0001_stage1_register_quality_gate/register_similarity_images512/cache_jobs.json",
    )
    parser.add_argument(
        "--cache-root",
        default="caches/vggt_omega/0001_stage1_register_quality_gate/native_geometry_images512",
    )
    parser.add_argument(
        "--analysis-dir",
        default="runs/0001_stage1_register_quality_gate/vggt_native_geometry_images512",
    )
    parser.add_argument(
        "--doc-output-dir",
        default="docs/experiments/0001_stage1_register_quality_gate/vggt_native_geometry",
    )
    parser.add_argument(
        "--register-similarity-csv",
        default="docs/experiments/0001_stage1_register_quality_gate/register_similarity/subset_register_similarity.csv",
    )
    parser.add_argument("--checkpoint", default="512")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-cache-jobs", type=int, default=None)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--max-pixels-per-image", type=int, default=1024)
    parser.add_argument("--max-pointmap-points", type=int, default=60000)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args(argv)

    project_root = Path.cwd()
    cache_root = project_root / args.cache_root
    analysis_dir = project_root / args.analysis_dir
    doc_output_dir = project_root / args.doc_output_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    scene_filter = set(args.scene)
    cache_jobs, records = build_cache_jobs(
        source_jobs_path=project_root / args.source_cache_jobs,
        cache_root=cache_root,
        scene_filter=scene_filter,
    )
    if not cache_jobs:
        raise SystemExit("No cache jobs matched.")
    cache_jobs_path = analysis_dir / "cache_jobs.json"
    cache_jobs_path.write_text(json.dumps(cache_jobs, indent=2) + "\n", encoding="utf-8")

    if not args.skip_cache:
        jobs_for_run = cache_jobs[: args.max_cache_jobs] if args.max_cache_jobs else cache_jobs
        run_vggt_batch_cache(
            jobs_json=write_limited_jobs(analysis_dir, jobs_for_run),
            checkpoint=args.checkpoint,
            image_resolution=args.image_resolution,
            mode=args.mode,
            device=args.device,
            force=args.force_cache,
        )
    if args.cache_only:
        print(json.dumps({"cache_jobs": len(cache_jobs[: args.max_cache_jobs] if args.max_cache_jobs else cache_jobs)}))
        return 0

    register_rows = load_register_similarity(project_root / args.register_similarity_csv)
    subset_rows = analyze_native_consistency(
        records=records,
        register_rows=register_rows,
        max_pixels_per_image=args.max_pixels_per_image,
        max_pointmap_points=args.max_pointmap_points,
        epsilon=args.epsilon,
    )
    scene_rows = build_scene_correlations(subset_rows)
    correlation_summary_rows = summarize_correlations(scene_rows)
    dataset_rows = summarize_by_dataset(subset_rows)
    summary = {
        "subset_rows": len(subset_rows),
        "scene_correlation_rows": len(scene_rows),
        "correlation_summary_rows": len(correlation_summary_rows),
        "dataset_summary_rows": len(dataset_rows),
        "cache_jobs": len(cache_jobs),
        "max_pixels_per_image": args.max_pixels_per_image,
        "max_pointmap_points": args.max_pointmap_points,
        "reference": "full_train_non_test VGGT-Omega depth/pose cache",
        "notes": [
            "Each subset image is compared against the same image in the full-train(non-test) cache.",
            "Depth is aligned by a per-image median scale before abs-rel and log-RMSE are computed.",
            "Pose and point-map metrics use Umeyama similarity alignment to remove global frame and scale ambiguity.",
            "Point-map is derived from VGGT-Omega depth plus decoded pose/intrinsics; this local model does not emit a direct point-map head.",
        ],
    }
    for output_dir in (analysis_dir, doc_output_dir):
        write_csv(output_dir / "native_subset_consistency.csv", subset_rows)
        write_csv(output_dir / "native_scene_correlations.csv", scene_rows)
        write_csv(output_dir / "native_correlation_summary.csv", correlation_summary_rows)
        write_csv(output_dir / "native_dataset_summary.csv", dataset_rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


def build_cache_jobs(
    *,
    source_jobs_path: Path,
    cache_root: Path,
    scene_filter: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_jobs = json.loads(source_jobs_path.read_text(encoding="utf-8"))
    cache_jobs = []
    records = []
    for job in source_jobs:
        scene_id = str(job["scene_id"])
        if scene_filter and scene_id not in scene_filter:
            continue
        method = str(job["method"])
        output_dir = cache_root / scene_id / method
        new_job = {
            **job,
            "output_dir": str(output_dir.resolve()),
        }
        cache_jobs.append(new_job)
        records.append(
            {
                "scene_id": scene_id,
                "method": method,
                "role": str(job["role"]),
                "image_count": int(job["image_count"]),
                "image_list": str(job["image_list"]),
                "cache_dir": str(output_dir.resolve()),
            }
        )
    return cache_jobs, records


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


def write_limited_jobs(analysis_dir: Path, jobs: list[dict[str, Any]]) -> Path:
    path = analysis_dir / "cache_jobs_to_run.json"
    path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return path


def load_register_similarity(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(row["scene_id"], row["method"])] = {
                "image_count": float(row["image_count"]),
                "register_mean_cosine": float(row["register_mean_cosine"]),
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "lpips": float(row["lpips"]),
            }
    return rows


def analyze_native_consistency(
    *,
    records: list[dict[str, Any]],
    register_rows: dict[tuple[str, str], dict[str, float]],
    max_pixels_per_image: int,
    max_pointmap_points: int,
    epsilon: float,
) -> list[dict[str, Any]]:
    import torch
    from vggt_omega.utils.pose_enc import encoding_to_camera

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
            raise RuntimeError(f"{scene_id}: missing full_train_non_test reference")
        ref_cache = load_native_cache(Path(reference["cache_dir"]), torch)
        ref_image_index = {normalize_image_key(path): index for index, path in enumerate(ref_cache["images"])}

        for subset in sorted(subsets_by_scene[scene_id], key=lambda item: method_order(short_method(item["method"]))):
            method = short_method(subset["method"])
            similarity = register_rows.get((scene_id, method))
            if similarity is None:
                raise RuntimeError(f"{scene_id}/{method}: missing register similarity")
            sub_cache = load_native_cache(Path(subset["cache_dir"]), torch)
            pairs = pair_indices(ref_image_index, sub_cache["images"])
            if not pairs:
                raise RuntimeError(f"{scene_id}/{method}: no common images with full reference")

            ref_indices = [left for left, _right in pairs]
            sub_indices = [right for _left, right in pairs]
            ref_depth = ref_cache["depth"][0, ref_indices, ..., 0].float()
            sub_depth = sub_cache["depth"][0, sub_indices, ..., 0].float()
            ref_conf = ref_cache["depth_conf"][0, ref_indices].float()
            sub_conf = sub_cache["depth_conf"][0, sub_indices].float()
            ref_pose = ref_cache["pose_enc"][0, ref_indices].float()
            sub_pose = sub_cache["pose_enc"][0, sub_indices].float()

            depth_metrics, sampled_pixels = compare_depth(
                scene_id=scene_id,
                method=method,
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

            pose_metrics, alignment = compare_pose(ref_extrinsics, sub_extrinsics, ref_pose, sub_pose)
            point_metrics = compare_pointmaps(
                scene_id=scene_id,
                method=method,
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

            row = {
                "scene_id": scene_id,
                "dataset": dataset_group(scene_id),
                "method": method,
                "image_count": int(similarity["image_count"]),
                "common_images": len(pairs),
                "sampled_depth_pixels": sampled_pixels,
                "sampled_pointmap_points": int(point_metrics.pop("sampled_pointmap_points")),
                "register_mean_cosine": round(similarity["register_mean_cosine"], 6),
                "psnr": round(similarity["psnr"], 6),
                "ssim": round(similarity["ssim"], 6),
                "lpips": round(similarity["lpips"], 6),
                **round_metric_dict(drop_internal(depth_metrics)),
                **round_metric_dict(pose_metrics),
                **round_metric_dict(point_metrics),
                "alignment_scale": round(float(alignment["scale"]), 8),
            }
            rows.append(row)
    return rows


def load_native_cache(cache_dir: Path, torch: Any) -> dict[str, Any]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    required = ["depth.pt", "depth_conf.pt", "pose_enc.pt"]
    for filename in required:
        if not (cache_dir / filename).exists():
            raise RuntimeError(f"{cache_dir}: missing {filename}")
    return {
        "images": manifest["images"],
        "depth": torch.load(cache_dir / "depth.pt", map_location="cpu"),
        "depth_conf": torch.load(cache_dir / "depth_conf.pt", map_location="cpu"),
        "pose_enc": torch.load(cache_dir / "pose_enc.pt", map_location="cpu"),
    }


def normalize_image_key(path: str) -> str:
    return str(Path(path).resolve())


def pair_indices(ref_index: dict[str, int], subset_images: list[str]) -> list[tuple[int, int]]:
    pairs = []
    for subset_index, image_path in enumerate(subset_images):
        key = normalize_image_key(image_path)
        if key in ref_index:
            pairs.append((ref_index[key], subset_index))
    return pairs


def compare_depth(
    *,
    scene_id: str,
    method: str,
    ref_depth: Any,
    sub_depth: Any,
    ref_conf: Any,
    sub_conf: Any,
    max_pixels_per_image: int,
    epsilon: float,
    torch: Any,
) -> tuple[dict[str, Any], int]:
    abs_rel_values = []
    log_values = []
    valid_scales = []
    scales_by_frame = [1.0] * int(ref_depth.shape[0])
    conf_values = []
    for frame_index in range(ref_depth.shape[0]):
        h, w = ref_depth.shape[-2:]
        indices = sample_flat_indices(
            count=h * w,
            max_count=max_pixels_per_image,
            seed_key=f"{scene_id}:{method}:depth:{frame_index}",
        )
        ys = torch.as_tensor(indices // w, dtype=torch.long)
        xs = torch.as_tensor(indices % w, dtype=torch.long)
        ref = ref_depth[frame_index, ys, xs]
        sub = sub_depth[frame_index, ys, xs]
        conf = torch.sqrt(torch.clamp(ref_conf[frame_index, ys, xs], min=0) * torch.clamp(sub_conf[frame_index, ys, xs], min=0))
        mask = torch.isfinite(ref) & torch.isfinite(sub) & (ref > epsilon) & (sub > epsilon)
        if int(mask.sum()) < 8:
            continue
        ref = ref[mask]
        sub = sub[mask]
        conf = conf[mask]
        scale = torch.median(ref / torch.clamp(sub, min=epsilon))
        scales_by_frame[frame_index] = float(scale)
        aligned = sub * scale
        abs_rel = torch.abs(aligned - ref) / torch.clamp(torch.abs(ref), min=epsilon)
        log_error = torch.log(torch.clamp(aligned, min=epsilon)) - torch.log(torch.clamp(ref, min=epsilon))
        abs_rel_values.append(abs_rel.cpu().numpy())
        log_values.append(log_error.cpu().numpy())
        valid_scales.append(float(scale))
        conf_values.append(float(torch.mean(conf)))
    if not abs_rel_values:
        raise RuntimeError(f"{scene_id}/{method}: no valid depth samples")
    abs_rel_all = np.concatenate(abs_rel_values)
    log_all = np.concatenate(log_values)
    scale_array = np.asarray(valid_scales, dtype=np.float64)
    return (
        {
            "depth_absrel_mean": float(np.mean(abs_rel_all)),
            "depth_absrel_median": float(np.median(abs_rel_all)),
            "depth_log_rmse": float(np.sqrt(np.mean(np.square(log_all)))),
            "depth_scale_median": float(np.median(scale_array)),
            "depth_scale_mad": float(np.median(np.abs(scale_array - np.median(scale_array)))),
            "depth_conf_mean": float(np.mean(conf_values)),
            "per_image_depth_scale": scales_by_frame,
        },
        int(len(abs_rel_all)),
    )


def compare_pose(ref_extrinsics: Any, sub_extrinsics: Any, ref_pose: Any, sub_pose: Any) -> tuple[dict[str, float], dict[str, Any]]:
    ref_centers = camera_centers(ref_extrinsics).numpy()
    sub_centers = camera_centers(sub_extrinsics).numpy()
    alignment = umeyama_alignment(sub_centers, ref_centers)
    sub_centers_aligned = apply_similarity(sub_centers, alignment)
    scale = bbox_diagonal(ref_centers)
    center_errors = np.linalg.norm(sub_centers_aligned - ref_centers, axis=1) / scale

    ref_rot_c2w = np.transpose(ref_extrinsics[:, :3, :3].numpy(), (0, 2, 1))
    sub_rot_c2w = np.transpose(sub_extrinsics[:, :3, :3].numpy(), (0, 2, 1))
    align_rot = alignment["rotation"]
    rotation_errors = []
    for ref_rot, sub_rot in zip(ref_rot_c2w, sub_rot_c2w, strict=True):
        sub_aligned = align_rot @ sub_rot
        delta = sub_aligned @ ref_rot.T
        trace_value = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(math.degrees(math.acos(trace_value)))
    fov_error = np.abs(sub_pose[:, 7:9].numpy() - ref_pose[:, 7:9].numpy())
    return (
        {
            "pose_center_rmse_norm": float(np.sqrt(np.mean(np.square(center_errors)))),
            "pose_center_median_norm": float(np.median(center_errors)),
            "pose_center_p90_norm": float(np.quantile(center_errors, 0.90)),
            "pose_rotation_mean_deg": float(np.mean(rotation_errors)),
            "pose_rotation_median_deg": float(np.median(rotation_errors)),
            "pose_fov_abs_mean": float(np.mean(fov_error)),
        },
        alignment,
    )


def compare_pointmaps(
    *,
    scene_id: str,
    method: str,
    ref_depth: Any,
    sub_depth: Any,
    ref_extrinsics: Any,
    sub_extrinsics: Any,
    ref_intrinsics: Any,
    sub_intrinsics: Any,
    depth_scales: list[float],
    max_pixels_per_image: int,
    max_pointmap_points: int,
    epsilon: float,
    torch: Any,
) -> dict[str, float]:
    ref_points_chunks = []
    sub_points_chunks = []
    for frame_index in range(ref_depth.shape[0]):
        h, w = ref_depth.shape[-2:]
        indices = sample_flat_indices(
            count=h * w,
            max_count=max_pixels_per_image,
            seed_key=f"{scene_id}:{method}:pointmap:{frame_index}",
        )
        ys = torch.as_tensor(indices // w, dtype=torch.long)
        xs = torch.as_tensor(indices % w, dtype=torch.long)
        ref_z = ref_depth[frame_index, ys, xs]
        sub_z = sub_depth[frame_index, ys, xs] * float(depth_scales[min(frame_index, len(depth_scales) - 1)])
        mask = torch.isfinite(ref_z) & torch.isfinite(sub_z) & (ref_z > epsilon) & (sub_z > epsilon)
        if int(mask.sum()) < 8:
            continue
        xs = xs[mask].float()
        ys = ys[mask].float()
        ref_z = ref_z[mask].float()
        sub_z = sub_z[mask].float()
        ref_points_chunks.append(
            pixels_to_world(xs, ys, ref_z, ref_intrinsics[frame_index], ref_extrinsics[frame_index], torch)
        )
        sub_points_chunks.append(
            pixels_to_world(xs, ys, sub_z, sub_intrinsics[frame_index], sub_extrinsics[frame_index], torch)
        )
    if not ref_points_chunks:
        raise RuntimeError(f"{scene_id}/{method}: no valid point-map samples")
    ref_points = torch.cat(ref_points_chunks, dim=0).numpy()
    sub_points = torch.cat(sub_points_chunks, dim=0).numpy()
    if len(ref_points) > max_pointmap_points:
        indices = sample_flat_indices(
            count=len(ref_points),
            max_count=max_pointmap_points,
            seed_key=f"{scene_id}:{method}:pointmap:global",
        )
        ref_points = ref_points[indices]
        sub_points = sub_points[indices]
    alignment = umeyama_alignment(sub_points, ref_points)
    sub_aligned = apply_similarity(sub_points, alignment)
    scale = bbox_diagonal(ref_points)
    distances = np.linalg.norm(sub_aligned - ref_points, axis=1) / scale
    return {
        "sampled_pointmap_points": float(len(distances)),
        "pointmap_l1_norm": float(np.mean(distances)),
        "pointmap_rmse_norm": float(np.sqrt(np.mean(np.square(distances)))),
        "pointmap_median_norm": float(np.median(distances)),
        "pointmap_p90_norm": float(np.quantile(distances, 0.90)),
    }


def pixels_to_world(xs: Any, ys: Any, depth: Any, intrinsics: Any, extrinsics: Any, torch: Any) -> Any:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x_cam = (xs - cx) / fx * depth
    y_cam = (ys - cy) / fy * depth
    cam_points = torch.stack([x_cam, y_cam, depth], dim=1)
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    return (cam_points - translation[None]) @ rotation


def camera_centers(extrinsics: Any) -> Any:
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return -torch_matmul_transpose(rotation, translation)


def torch_matmul_transpose(rotation: Any, translation: Any) -> Any:
    return rotation.transpose(1, 2).matmul(translation[..., None]).squeeze(-1)


def umeyama_alignment(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        singular_values[-1] *= -1
        rotation = vt.T @ u.T
    variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.sum(singular_values) / max(variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return {"scale": scale, "rotation": rotation, "translation": translation}


def apply_similarity(points: np.ndarray, alignment: dict[str, Any]) -> np.ndarray:
    return alignment["scale"] * (points @ alignment["rotation"].T) + alignment["translation"]


def bbox_diagonal(points: np.ndarray) -> float:
    diagonal = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    return diagonal if diagonal > 1e-12 else 1.0


def sample_flat_indices(*, count: int, max_count: int, seed_key: str) -> np.ndarray:
    if count <= max_count:
        return np.arange(count, dtype=np.int64)
    seed = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_count, replace=False)).astype(np.int64)


def build_scene_correlations(subset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        name
        for name in subset_rows[0]
        if name.startswith(ERROR_METRIC_PREFIXES)
        and name
        not in {
            "depth_scale_median",
            "depth_scale_mad",
            "depth_conf_mean",
        }
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset_rows:
        grouped[row["scene_id"]].append(row)
    scene_rows = []
    for scene_id, rows in sorted(grouped.items()):
        for metric in metric_names:
            cosine_values = [float(row["register_mean_cosine"]) for row in rows]
            metric_values = [float(row[metric]) for row in rows]
            best_metric = min(rows, key=lambda row: float(row[metric]))
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "dataset": rows[0]["dataset"],
                    "metric": metric,
                    "expected_direction": "negative",
                    "n": len(rows),
                    "spearman": round(spearman(cosine_values, metric_values), 6),
                    "pearson": round(pearson(cosine_values, metric_values), 6),
                    "best_cosine_method": max(rows, key=lambda row: float(row["register_mean_cosine"]))["method"],
                    "best_metric_method": best_metric["method"],
                }
            )
    return scene_rows


def summarize_correlations(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        grouped[row["metric"]].append(row)
    summary_rows = []
    for metric, rows in sorted(grouped.items()):
        spearman_values = [float(row["spearman"]) for row in rows]
        pearson_values = [float(row["pearson"]) for row in rows]
        summary_rows.append(
            {
                "metric": metric,
                "expected_direction": "negative",
                "scenes": len(rows),
                "mean_spearman": round(sum(spearman_values) / len(spearman_values), 6),
                "spearman_expected_sign": f"{sum(value < 0 for value in spearman_values)}/{len(rows)}",
                "mean_pearson": round(sum(pearson_values) / len(pearson_values), 6),
                "pearson_expected_sign": f"{sum(value < 0 for value in pearson_values)}/{len(rows)}",
                "best_match": f"{sum(row['best_cosine_method'] == row['best_metric_method'] for row in rows)}/{len(rows)}",
            }
        )
    return summary_rows


def summarize_by_dataset(subset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        name
        for name in subset_rows[0]
        if name.startswith(ERROR_METRIC_PREFIXES)
        and name not in {"depth_scale_median", "depth_scale_mad", "depth_conf_mean"}
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subset_rows:
        grouped[(row["dataset"], row["method"])].append(row)
    rows = []
    for (dataset, method), items in sorted(grouped.items()):
        row: dict[str, Any] = {"dataset": dataset, "method": method, "runs": len(items)}
        for metric in metric_names:
            row[f"{metric}_mean"] = round(sum(float(item[metric]) for item in items) / len(items), 8)
        rows.append(row)
    return rows


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    if left_var == 0 or right_var == 0:
        return float("nan")
    return numerator / math.sqrt(left_var * right_var)


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rank(left), rank(right))


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = average_rank
        cursor = end
    return ranks


def drop_internal(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "per_image_depth_scale"}


def round_metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: round(float(value), 8) for key, value in metrics.items()}


def short_method(method: str) -> str:
    return method.replace("random_ratio_", "random_").replace("uniform_stride_ratio", "uniform")


def method_order(method: str) -> int:
    if method.startswith("random_seed"):
        long_method = method.replace("random_seed", "random_ratio_seed", 1)
    elif method == "uniform":
        long_method = "uniform_stride_ratio"
    else:
        long_method = method
    try:
        return METHODS.index(long_method)
    except ValueError:
        return len(METHODS)


def dataset_group(scene_id: str) -> str:
    if scene_id.startswith("mipnerf360_"):
        return "mipnerf360"
    if scene_id.startswith("tandt_"):
        return "tandt"
    if scene_id.startswith("db_"):
        return "db"
    return "unknown"


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


if __name__ == "__main__":
    raise SystemExit(main())
