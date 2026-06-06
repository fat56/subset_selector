#!/usr/bin/env python3
"""Compute Stage 1 point-cloud geometry proxies against register similarity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vggt_omega_selector.data.registry import load_scene_records


METHODS = [f"random_ratio_seed{seed:03d}" for seed in range(5)] + ["uniform_stride_ratio"]
LOWER_IS_BETTER_PREFIXES = ("accuracy_", "completeness_", "chamfer_")
PLY_NUMPY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class GeometryRecord:
    scene_id: str
    dataset_group: str
    method: str
    image_count: int
    point_cloud_path: Path
    register_mean_cosine: float
    psnr: float
    ssim: float
    lpips: float


@dataclass(frozen=True)
class ReferenceRecord:
    scene_id: str
    reference_type: str
    reference_path: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 1 geometry proxy metrics.")
    parser.add_argument(
        "--jobs-tsv",
        default="runs/0001_stage1_register_quality_gate/queues/random_uniform_images4_30k/jobs.tsv",
    )
    parser.add_argument("--dataset-registry", default="data/datasets.yaml")
    parser.add_argument(
        "--register-similarity-csv",
        default="docs/experiments/0001_stage1_register_quality_gate/register_similarity/subset_register_similarity.csv",
    )
    parser.add_argument(
        "--doc-output-dir",
        default="docs/experiments/0001_stage1_register_quality_gate/geometry_metrics",
    )
    parser.add_argument(
        "--analysis-dir",
        default="runs/0001_stage1_register_quality_gate/geometry_metrics",
    )
    parser.add_argument(
        "--full-fastgs-root",
        default="runs/0001_stage1_register_quality_gate/fastgs_full_train/3dgsdata",
    )
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N when torch is available.")
    parser.add_argument("--tau", type=float, action="append", default=[0.005, 0.01, 0.02])
    parser.add_argument("--scene", action="append", default=[])
    args = parser.parse_args(argv)

    project_root = Path.cwd()
    doc_output_dir = project_root / args.doc_output_dir
    analysis_dir = project_root / args.analysis_dir
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    register_rows = load_register_similarity(project_root / args.register_similarity_csv)
    run_records = build_geometry_records(
        jobs_tsv=project_root / args.jobs_tsv,
        register_rows=register_rows,
        iteration=args.iteration,
        scene_filter=set(args.scene),
    )
    scene_records = {
        scene.scene_id: scene for scene in load_scene_records(project_root / args.dataset_registry, project_root=project_root)
    }
    references = build_reference_records(
        scene_records=scene_records,
        full_fastgs_root=project_root / args.full_fastgs_root,
        scene_filter=set(args.scene),
        iteration=args.iteration,
    )
    if not references:
        raise SystemExit("No geometry references found.")

    backend = DistanceBackend(args.device)
    subset_rows = analyze_geometry(
        run_records=run_records,
        references=references,
        max_points=args.max_points,
        batch_size=args.batch_size,
        tau_values=args.tau,
        backend=backend,
    )
    scene_rows = build_scene_correlations(subset_rows)
    correlation_summary_rows = summarize_correlations(scene_rows)
    dataset_rows = summarize_by_dataset(subset_rows)
    summary = {
        "subset_rows": len(subset_rows),
        "scene_correlation_rows": len(scene_rows),
        "correlation_summary_rows": len(correlation_summary_rows),
        "dataset_summary_rows": len(dataset_rows),
        "references": [
            {
                "scene_id": reference.scene_id,
                "reference_type": reference.reference_type,
                "reference_path": str(reference.reference_path),
            }
            for reference in references
        ],
        "max_points": args.max_points,
        "tau": args.tau,
        "distance_backend": backend.name,
        "notes": [
            "Distances are normalized by the reference point cloud bounding-box diagonal.",
            "colmap_sparse_full_scene uses the original full-scene COLMAP sparse points3D.ply as pseudo-GT.",
            "fastgs_full_train is only emitted for scenes with a completed full-train FastGS point cloud.",
            "FastGS point clouds are raw Gaussian centers, not fused surface samples.",
        ],
    }

    for output_dir in (doc_output_dir, analysis_dir):
        write_csv(output_dir / "geometry_subset_metrics.csv", subset_rows)
        write_csv(output_dir / "geometry_scene_correlations.csv", scene_rows)
        write_csv(output_dir / "geometry_correlation_summary.csv", correlation_summary_rows)
        write_csv(output_dir / "geometry_dataset_summary.csv", dataset_rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


def load_register_similarity(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
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


def build_geometry_records(
    *,
    jobs_tsv: Path,
    register_rows: dict[tuple[str, str], dict[str, float]],
    iteration: int,
    scene_filter: set[str],
) -> list[GeometryRecord]:
    records = []
    with jobs_tsv.open(encoding="utf-8") as handle:
        for job in csv.DictReader(handle, delimiter="\t"):
            scene_id = job["scene_id"]
            if scene_filter and scene_id not in scene_filter:
                continue
            method = job["method"]
            short = short_method(method)
            point_cloud_path = (
                Path(job["model_path"]) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
            )
            if not point_cloud_path.exists():
                raise RuntimeError(f"Missing point cloud for {scene_id}/{method}: {point_cloud_path}")
            similarity = register_rows.get((scene_id, short))
            if similarity is None:
                raise RuntimeError(f"Missing register similarity row for {scene_id}/{short}")
            records.append(
                GeometryRecord(
                    scene_id=scene_id,
                    dataset_group=dataset_group(scene_id),
                    method=short,
                    image_count=int(similarity.get("image_count", 0) or 0),
                    point_cloud_path=point_cloud_path,
                    register_mean_cosine=similarity["register_mean_cosine"],
                    psnr=similarity["psnr"],
                    ssim=similarity["ssim"],
                    lpips=similarity["lpips"],
                )
            )
    return sorted(records, key=lambda item: (item.scene_id, method_order(item.method)))


def build_reference_records(
    *,
    scene_records: dict[str, Any],
    full_fastgs_root: Path,
    scene_filter: set[str],
    iteration: int,
) -> list[ReferenceRecord]:
    references = []
    for scene_id, scene in sorted(scene_records.items()):
        if scene_filter and scene_id not in scene_filter:
            continue
        colmap_reference = scene.sparse_path / "points3D.ply"
        if colmap_reference.exists():
            references.append(
                ReferenceRecord(
                    scene_id=scene_id,
                    reference_type="colmap_sparse_full_scene",
                    reference_path=colmap_reference,
                )
            )

        full_candidates = sorted(
            (full_fastgs_root / scene_id).glob(
                f"*/model/point_cloud/iteration_{iteration}/point_cloud.ply"
            )
        )
        image4_candidates = [
            candidate for candidate in full_candidates if "images4" in candidate.as_posix()
        ]
        selected = image4_candidates[0] if image4_candidates else None
        if selected is not None:
            references.append(
                ReferenceRecord(
                    scene_id=scene_id,
                    reference_type="fastgs_full_train_images4",
                    reference_path=selected,
                )
            )
    return references


def analyze_geometry(
    *,
    run_records: list[GeometryRecord],
    references: list[ReferenceRecord],
    max_points: int,
    batch_size: int,
    tau_values: list[float],
    backend: "DistanceBackend",
) -> list[dict[str, Any]]:
    by_scene: dict[str, list[GeometryRecord]] = defaultdict(list)
    for record in run_records:
        by_scene[record.scene_id].append(record)
    references_by_scene: dict[str, list[ReferenceRecord]] = defaultdict(list)
    for reference in references:
        references_by_scene[reference.scene_id].append(reference)

    rows: list[dict[str, Any]] = []
    point_cache: dict[Path, np.ndarray] = {}
    for scene_id in sorted(by_scene):
        for reference in references_by_scene.get(scene_id, []):
            reference_points_raw = load_points_cached(reference.reference_path, point_cache)
            reference_points = sample_points(
                reference_points_raw,
                max_points=max_points,
                seed_key=f"{scene_id}:{reference.reference_type}:reference",
            )
            scale = bbox_diagonal(reference_points)
            if scale <= 0:
                raise RuntimeError(f"{reference.reference_path}: invalid reference scale")

            for record in sorted(by_scene[scene_id], key=lambda item: method_order(item.method)):
                subset_points_raw = load_points_cached(record.point_cloud_path, point_cache)
                subset_points = sample_points(
                    subset_points_raw,
                    max_points=max_points,
                    seed_key=f"{scene_id}:{record.method}:subset",
                )
                subset_to_ref = backend.min_distances(
                    query=subset_points,
                    target=reference_points,
                    batch_size=batch_size,
                )
                ref_to_subset = backend.min_distances(
                    query=reference_points,
                    target=subset_points,
                    batch_size=batch_size,
                )
                rows.append(
                    metric_row(
                        record=record,
                        reference=reference,
                        reference_points_raw=reference_points_raw,
                        reference_points=reference_points,
                        subset_points_raw=subset_points_raw,
                        subset_points=subset_points,
                        subset_to_ref=subset_to_ref,
                        ref_to_subset=ref_to_subset,
                        scale=scale,
                        tau_values=tau_values,
                    )
                )
    return rows


def metric_row(
    *,
    record: GeometryRecord,
    reference: ReferenceRecord,
    reference_points_raw: np.ndarray,
    reference_points: np.ndarray,
    subset_points_raw: np.ndarray,
    subset_points: np.ndarray,
    subset_to_ref: np.ndarray,
    ref_to_subset: np.ndarray,
    scale: float,
    tau_values: list[float],
) -> dict[str, Any]:
    accuracy = float(np.mean(subset_to_ref) / scale)
    completeness = float(np.mean(ref_to_subset) / scale)
    accuracy_p90 = float(np.quantile(subset_to_ref, 0.90) / scale)
    completeness_p90 = float(np.quantile(ref_to_subset, 0.90) / scale)
    chamfer_l1 = 0.5 * (accuracy + completeness)
    chamfer_l2_sq = 0.5 * (
        float(np.mean(np.square(subset_to_ref)) / (scale * scale))
        + float(np.mean(np.square(ref_to_subset)) / (scale * scale))
    )
    row: dict[str, Any] = {
        "scene_id": record.scene_id,
        "dataset": record.dataset_group,
        "method": record.method,
        "image_count": record.image_count,
        "reference_type": reference.reference_type,
        "register_mean_cosine": round(record.register_mean_cosine, 6),
        "psnr": round(record.psnr, 6),
        "ssim": round(record.ssim, 6),
        "lpips": round(record.lpips, 6),
        "reference_scale": round(scale, 6),
        "subset_points_raw": int(len(subset_points_raw)),
        "reference_points_raw": int(len(reference_points_raw)),
        "subset_points_sampled": int(len(subset_points)),
        "reference_points_sampled": int(len(reference_points)),
        "accuracy_mean_norm": round(accuracy, 8),
        "accuracy_p90_norm": round(accuracy_p90, 8),
        "completeness_mean_norm": round(completeness, 8),
        "completeness_p90_norm": round(completeness_p90, 8),
        "chamfer_l1_norm": round(chamfer_l1, 8),
        "chamfer_l2_norm_sq": round(chamfer_l2_sq, 10),
    }
    for tau in tau_values:
        tau_name = tau_slug(tau)
        tau_abs = tau * scale
        precision = float(np.mean(subset_to_ref <= tau_abs))
        recall = float(np.mean(ref_to_subset <= tau_abs))
        fscore = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        row[f"precision_tau{tau_name}"] = round(precision, 8)
        row[f"recall_tau{tau_name}"] = round(recall, 8)
        row[f"fscore_tau{tau_name}"] = round(fscore, 8)
    return row


def build_scene_correlations(subset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        name
        for name in subset_rows[0]
        if name.startswith(("accuracy_", "completeness_", "chamfer_", "precision_", "recall_", "fscore_"))
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subset_rows:
        grouped[(row["scene_id"], row["reference_type"])].append(row)

    scene_rows = []
    for (scene_id, reference_type), rows in sorted(grouped.items()):
        for metric in metric_names:
            cosine_values = [float(row["register_mean_cosine"]) for row in rows]
            metric_values = [float(row[metric]) for row in rows]
            lower_is_better = metric.startswith(LOWER_IS_BETTER_PREFIXES)
            best_metric = min(rows, key=lambda row: float(row[metric])) if lower_is_better else max(
                rows, key=lambda row: float(row[metric])
            )
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "dataset": rows[0]["dataset"],
                    "reference_type": reference_type,
                    "metric": metric,
                    "expected_direction": "negative" if lower_is_better else "positive",
                    "n": len(rows),
                    "spearman": round(spearman(cosine_values, metric_values), 6),
                    "pearson": round(pearson(cosine_values, metric_values), 6),
                    "best_cosine_method": max(rows, key=lambda row: float(row["register_mean_cosine"]))["method"],
                    "best_metric_method": best_metric["method"],
                }
            )
    return scene_rows


def summarize_by_dataset(subset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        name
        for name in subset_rows[0]
        if name.startswith(("accuracy_", "completeness_", "chamfer_", "precision_", "recall_", "fscore_"))
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subset_rows:
        grouped[(row["dataset"], row["reference_type"], row["method"])].append(row)
    summary_rows = []
    for (dataset, reference_type, method), rows in sorted(grouped.items()):
        row: dict[str, Any] = {
            "dataset": dataset,
            "reference_type": reference_type,
            "method": method,
            "runs": len(rows),
        }
        for metric in metric_names:
            row[f"{metric}_mean"] = round(sum(float(item[metric]) for item in rows) / len(rows), 8)
        summary_rows.append(row)
    return summary_rows


def summarize_correlations(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        grouped[(row["reference_type"], row["metric"])].append(row)
    summary_rows = []
    for (reference_type, metric), rows in sorted(grouped.items()):
        direction = rows[0]["expected_direction"]
        spearman_values = [float(row["spearman"]) for row in rows]
        pearson_values = [float(row["pearson"]) for row in rows]
        if direction == "negative":
            spearman_sign = sum(value < 0 for value in spearman_values)
            pearson_sign = sum(value < 0 for value in pearson_values)
        else:
            spearman_sign = sum(value > 0 for value in spearman_values)
            pearson_sign = sum(value > 0 for value in pearson_values)
        summary_rows.append(
            {
                "reference_type": reference_type,
                "metric": metric,
                "expected_direction": direction,
                "scenes": len(rows),
                "mean_spearman": round(sum(spearman_values) / len(spearman_values), 6),
                "spearman_expected_sign": f"{spearman_sign}/{len(rows)}",
                "mean_pearson": round(sum(pearson_values) / len(pearson_values), 6),
                "pearson_expected_sign": f"{pearson_sign}/{len(rows)}",
                "best_match": f"{sum(row['best_cosine_method'] == row['best_metric_method'] for row in rows)}/{len(rows)}",
            }
        )
    return summary_rows


def load_points_cached(path: Path, cache: dict[Path, np.ndarray]) -> np.ndarray:
    resolved = path.resolve()
    cached = cache.get(resolved)
    if cached is not None:
        return cached
    points = load_ply_xyz(resolved)
    cache[resolved] = points
    return points


def load_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"{path}: missing PLY end_header")
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        fmt, vertex_count, properties = parse_ply_header(path, header_lines)
        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, PLY_NUMPY_TYPES[kind]) for kind, name in properties])
            data = np.fromfile(handle, dtype=dtype, count=vertex_count)
            points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32, copy=False)
        elif fmt == "ascii":
            rows = []
            x_index = next(index for index, (_kind, name) in enumerate(properties) if name == "x")
            y_index = next(index for index, (_kind, name) in enumerate(properties) if name == "y")
            z_index = next(index for index, (_kind, name) in enumerate(properties) if name == "z")
            for _ in range(vertex_count):
                values = handle.readline().decode("ascii", errors="replace").split()
                rows.append((float(values[x_index]), float(values[y_index]), float(values[z_index])))
            points = np.asarray(rows, dtype=np.float32)
        else:
            raise RuntimeError(f"{path}: unsupported PLY format {fmt}")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError(f"{path}: no finite xyz points")
    return points


def parse_ply_header(path: Path, header_lines: list[str]) -> tuple[str, int, list[tuple[str, str]]]:
    if not header_lines or header_lines[0] != "ply":
        raise RuntimeError(f"{path}: not a PLY file")
    fmt = ""
    vertex_count = None
    properties: list[tuple[str, str]] = []
    current_element = None
    for line in header_lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise RuntimeError(f"{path}: list vertex properties are not supported")
            kind = parts[1]
            name = parts[2]
            if kind not in PLY_NUMPY_TYPES:
                raise RuntimeError(f"{path}: unsupported PLY property type {kind}")
            properties.append((kind, name))
    if not fmt or vertex_count is None:
        raise RuntimeError(f"{path}: missing PLY format or vertex count")
    property_names = {name for _kind, name in properties}
    if not {"x", "y", "z"}.issubset(property_names):
        raise RuntimeError(f"{path}: PLY vertex properties must include x, y, z")
    return fmt, vertex_count, properties


def sample_points(points: np.ndarray, *, max_points: int, seed_key: str) -> np.ndarray:
    if len(points) <= max_points:
        return points.astype(np.float32, copy=False)
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    seed = struct.unpack("<Q", digest[:8])[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(indices)].astype(np.float32, copy=False)


def bbox_diagonal(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))


class DistanceBackend:
    def __init__(self, device: str) -> None:
        self.torch = None
        self.device = "numpy"
        try:
            import torch  # type: ignore
        except Exception:
            self.name = "numpy"
            return
        if device == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected = device
        self.torch = torch
        self.device = selected
        self.name = f"torch:{selected}"

    def min_distances(self, *, query: np.ndarray, target: np.ndarray, batch_size: int) -> np.ndarray:
        if self.torch is None:
            return min_distances_numpy(query=query, target=target, batch_size=batch_size)
        return min_distances_torch(
            query=query,
            target=target,
            batch_size=batch_size,
            torch_module=self.torch,
            device=self.device,
        )


def min_distances_torch(
    *,
    query: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    torch_module: Any,
    device: str,
) -> np.ndarray:
    torch = torch_module
    query_tensor = torch.as_tensor(query, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(query), batch_size):
            distances = torch.cdist(query_tensor[start : start + batch_size], target_tensor)
            chunks.append(torch.min(distances, dim=1).values.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def min_distances_numpy(*, query: np.ndarray, target: np.ndarray, batch_size: int) -> np.ndarray:
    target_batch_size = max(256, batch_size)
    best = np.full(len(query), np.inf, dtype=np.float32)
    for q_start in range(0, len(query), batch_size):
        q = query[q_start : q_start + batch_size]
        q_best = np.full(len(q), np.inf, dtype=np.float32)
        for t_start in range(0, len(target), target_batch_size):
            t = target[t_start : t_start + target_batch_size]
            distances_sq = np.sum(np.square(q[:, None, :] - t[None, :, :]), axis=2)
            q_best = np.minimum(q_best, np.min(distances_sq, axis=1))
        best[q_start : q_start + len(q)] = np.sqrt(q_best)
    return best


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


def tau_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


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
    raise SystemExit(main(sys.argv[1:]))
