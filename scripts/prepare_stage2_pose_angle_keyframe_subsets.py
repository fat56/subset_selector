#!/usr/bin/env python3
"""Build pose-angle keyframe subsets for experiment 0008."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FramePose:
    frame_id: str
    image_path: str
    index: int
    center: np.ndarray
    direction: np.ndarray
    azimuth: float
    elevation: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare pose-angle keyframe subsets from a scene manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--thresholds-deg", default="10,15,20")
    parser.add_argument("--uniform-anchor-count", type=int, default=5)
    parser.add_argument("--limit-scenes", type=int, default=None)
    args = parser.parse_args(argv)

    manifest_path = resolve(args.manifest)
    out_json = resolve(args.out_json)
    out_summary = resolve(args.out_summary)
    thresholds = parse_thresholds(args.thresholds_deg)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest["scenes"]
    if args.limit_scenes is not None:
        scenes = scenes[: args.limit_scenes]

    method_names = [f"pose_angle{int(threshold)}_keyframe{args.candidate_tag}" for threshold in thresholds]
    method_names.extend([f"pose_farthest_angle{args.candidate_tag}", f"pose_hybrid_uniform_angle{args.candidate_tag}"])

    scene_payloads = []
    for scene in scenes:
        scene_payloads.append(build_scene_payload(scene, args.candidate_tag, thresholds, args.uniform_anchor_count))

    output = {
        "experiment_id": "0008_stage2_pose_angle_keyframing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "candidate_tag": args.candidate_tag,
        "thresholds_deg": thresholds,
        "methods": ["uniform" + args.candidate_tag, *method_names],
        "scenes": scene_payloads,
        "summary": summarize(scene_payloads, ["uniform" + args.candidate_tag, *method_names]),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(render_summary(output), encoding="utf-8")
    print(json.dumps({"event": "pose_angle_subsets_done", **output["summary"]}, ensure_ascii=False), flush=True)
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_thresholds(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_scene_payload(
    scene: dict[str, Any],
    candidate_tag: str,
    thresholds: list[float],
    uniform_anchor_count: int,
) -> dict[str, Any]:
    frames = sorted(scene["frames"], key=lambda frame: int(frame.get("full_index", 0)))
    poses = load_frame_poses(frames)
    uniform_frame_ids = scene["splits"].get(f"uniform{candidate_tag}")
    if not uniform_frame_ids:
        raise ValueError(f"Scene {scene['scene_id']} has no uniform{candidate_tag} split.")
    frame_id_to_index = {pose.frame_id: pose.index for pose in poses}
    uniform_indices = sorted(frame_id_to_index[frame_id] for frame_id in uniform_frame_ids if frame_id in frame_id_to_index)
    if not uniform_indices:
        raise ValueError(f"Scene {scene['scene_id']} uniform{candidate_tag} split did not map to frames.")
    k = len(uniform_indices)

    directions = np.stack([pose.direction for pose in poses], axis=0)
    azimuths = np.array([pose.azimuth for pose in poses], dtype=np.float64)
    elevations = np.array([pose.elevation for pose in poses], dtype=np.float64)

    methods: dict[str, dict[str, Any]] = {}
    uniform_method = "uniform" + candidate_tag
    methods[uniform_method] = build_method_payload(uniform_method, uniform_indices, poses, directions, azimuths, elevations, uniform_indices)

    for threshold in thresholds:
        method = f"pose_angle{int(threshold)}_keyframe{candidate_tag}"
        selected = threshold_keyframes(azimuths, elevations, directions, math.radians(threshold), math.radians(threshold), k)
        methods[method] = build_method_payload(method, selected, poses, directions, azimuths, elevations, uniform_indices)

    farthest_method = f"pose_farthest_angle{candidate_tag}"
    farthest = farthest_angle_subset(directions, k, anchors=[0])
    methods[farthest_method] = build_method_payload(farthest_method, farthest, poses, directions, azimuths, elevations, uniform_indices)

    hybrid_method = f"pose_hybrid_uniform_angle{candidate_tag}"
    anchors = uniform_indices[: max(1, min(uniform_anchor_count, len(uniform_indices)))]
    hybrid = farthest_angle_subset(directions, k, anchors=anchors)
    methods[hybrid_method] = build_method_payload(hybrid_method, hybrid, poses, directions, azimuths, elevations, uniform_indices)

    return {
        "scene_id": scene["scene_id"],
        "scene_key": scene["scene_key"],
        "dataset": scene["dataset"],
        "full_count": len(poses),
        "k": k,
        "uniform_method": uniform_method,
        "frames": [
            {
                "index": pose.index,
                "frame_id": pose.frame_id,
                "image_path": pose.image_path,
                "azimuth_deg": round(math.degrees(pose.azimuth), 6),
                "elevation_deg": round(math.degrees(pose.elevation), 6),
            }
            for pose in poses
        ],
        "methods": methods,
    }


def load_frame_poses(frames: list[dict[str, Any]]) -> list[FramePose]:
    centers = []
    transforms = []
    for frame in frames:
        transform = load_transform(frame)
        transforms.append(transform)
        centers.append(transform[:3, 3].astype(np.float64))
    center_ref = np.median(np.stack(centers, axis=0), axis=0)

    poses = []
    for index, (frame, transform, center) in enumerate(zip(frames, transforms, centers, strict=True)):
        direction = center - center_ref
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            direction = -transform[:3, 2].astype(np.float64)
            norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            norm = 1.0
        direction = direction / norm
        y = float(np.clip(direction[1], -1.0, 1.0))
        poses.append(
            FramePose(
                frame_id=str(frame["frame_id"]),
                image_path=str(frame["image_path"]),
                index=index,
                center=center,
                direction=direction,
                azimuth=math.atan2(float(direction[0]), float(direction[2])),
                elevation=math.asin(y),
            )
        )
    return poses


def load_transform(frame: dict[str, Any]) -> np.ndarray:
    if frame.get("transform_matrix") is not None:
        return np.array(frame["transform_matrix"], dtype=np.float64).reshape(4, 4)
    pose_path = frame.get("pose_path")
    if not pose_path:
        raise ValueError(f"Frame {frame.get('frame_id')} has no transform_matrix or pose_path.")
    path = Path(pose_path)
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        for key in ("camera_pose", "transform_matrix", "c2w", "pose"):
            if key in payload:
                return np.array(payload[key], dtype=np.float64).reshape(4, 4)
    raise ValueError(f"Unsupported pose source for frame {frame.get('frame_id')}: {pose_path}")


def threshold_keyframes(
    azimuths: np.ndarray,
    elevations: np.ndarray,
    directions: np.ndarray,
    tau_phi: float,
    tau_theta: float,
    k: int,
) -> list[int]:
    selected = [0]
    for index in range(1, len(azimuths)):
        min_phi = min(circular_distance(float(azimuths[index]), float(azimuths[sel])) for sel in selected)
        min_theta = min(abs(float(elevations[index]) - float(elevations[sel])) for sel in selected)
        if min_phi > tau_phi or min_theta > tau_theta:
            selected.append(index)
    return adjust_to_k(selected, directions, k)


def adjust_to_k(selected: list[int], directions: np.ndarray, k: int) -> list[int]:
    selected = sorted(set(selected))
    if len(selected) == k:
        return selected
    if len(selected) > k:
        return farthest_angle_subset(directions, k, candidates=selected, anchors=[selected[0]])
    return farthest_angle_subset(directions, k, anchors=selected)


def farthest_angle_subset(
    directions: np.ndarray,
    k: int,
    *,
    anchors: list[int] | None = None,
    candidates: list[int] | None = None,
) -> list[int]:
    candidates = sorted(set(candidates if candidates is not None else range(len(directions))))
    selected = []
    for anchor in anchors or [candidates[0]]:
        if anchor in candidates and anchor not in selected:
            selected.append(anchor)
        if len(selected) >= k:
            return sorted(selected[:k])
    while len(selected) < k and len(selected) < len(candidates):
        best_index = None
        best_distance = -1.0
        for index in candidates:
            if index in selected:
                continue
            distance = min(spherical_angle(directions[index], directions[sel]) for sel in selected) if selected else math.pi
            if distance > best_distance or (math.isclose(distance, best_distance) and (best_index is None or index < best_index)):
                best_distance = distance
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
    return sorted(selected)


def build_method_payload(
    method: str,
    indices: list[int],
    poses: list[FramePose],
    directions: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    uniform_indices: list[int],
) -> dict[str, Any]:
    indices = sorted(indices)
    metrics = coverage_metrics(indices, directions, azimuths, elevations, uniform_indices)
    return {
        "method": method,
        "indices": indices,
        "frame_ids": [poses[index].frame_id for index in indices],
        "image_paths": [poses[index].image_path for index in indices],
        "metrics": metrics,
    }


def coverage_metrics(
    indices: list[int],
    directions: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    uniform_indices: list[int],
) -> dict[str, float]:
    pairs = [
        math.degrees(spherical_angle(directions[a], directions[b]))
        for pos, a in enumerate(indices)
        for b in indices[pos + 1 :]
    ]
    return {
        "selected_count": float(len(indices)),
        "azimuth_coverage_deg": circular_coverage_deg([float(azimuths[index]) for index in indices]),
        "elevation_coverage_deg": math.degrees(float(max(elevations[indices]) - min(elevations[indices]))) if indices else 0.0,
        "mean_pairwise_spherical_angle_deg": mean(pairs),
        "min_pairwise_spherical_angle_deg": min(pairs) if pairs else 0.0,
        "overlap_with_uniform20": len(set(indices) & set(uniform_indices)) / max(len(uniform_indices), 1),
        "temporal_span": float(max(indices) - min(indices)) if indices else 0.0,
    }


def circular_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def spherical_angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))


def circular_coverage_deg(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    ordered = sorted((value % (2.0 * math.pi)) for value in values)
    gaps = [ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)]
    gaps.append(ordered[0] + 2.0 * math.pi - ordered[-1])
    return math.degrees(2.0 * math.pi - max(gaps))


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def summarize(scene_payloads: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    by_method = {}
    by_dataset: dict[str, dict[str, Any]] = {}
    for method in methods:
        rows = [scene["methods"][method]["metrics"] for scene in scene_payloads if method in scene["methods"]]
        by_method[method] = summarize_metrics(rows)
        by_dataset[method] = {}
        datasets = sorted({scene["dataset"] for scene in scene_payloads})
        for dataset in datasets:
            dataset_rows = [
                scene["methods"][method]["metrics"]
                for scene in scene_payloads
                if scene["dataset"] == dataset and method in scene["methods"]
            ]
            by_dataset[method][dataset] = summarize_metrics(dataset_rows)
    return {
        "scene_count": len(scene_payloads),
        "dataset_counts": dataset_counts(scene_payloads),
        "by_method": by_method,
        "by_method_dataset": by_dataset,
    }


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        "selected_count",
        "azimuth_coverage_deg",
        "elevation_coverage_deg",
        "mean_pairwise_spherical_angle_deg",
        "min_pairwise_spherical_angle_deg",
        "overlap_with_uniform20",
        "temporal_span",
    ]
    return {key: mean([float(row[key]) for row in rows]) for key in keys}


def dataset_counts(scene_payloads: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scene in scene_payloads:
        counts[scene["dataset"]] = counts.get(scene["dataset"], 0) + 1
    return counts


def render_summary(output: dict[str, Any]) -> str:
    lines = [
        "# 0008 Pose-Angle Keyframe 离线诊断",
        "",
        f"- Manifest: `{output['manifest']}`",
        f"- 场景数: `{output['summary']['scene_count']}`",
        f"- 数据集分布: `{json.dumps(output['summary']['dataset_counts'], ensure_ascii=False)}`",
        f"- 阈值: `{output['thresholds_deg']}` degrees",
        "",
        "## 方法汇总",
        "",
        "| 方法 | 选帧数 | Azimuth 覆盖 | Elevation 覆盖 | Mean pair angle | Min pair angle | Overlap vs uniform | Temporal span |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in output["summary"]["by_method"].items():
        lines.append(
            "| {method} | {selected_count:.2f} | {azimuth_coverage_deg:.2f} | {elevation_coverage_deg:.2f} | "
            "{mean_pairwise_spherical_angle_deg:.2f} | {min_pairwise_spherical_angle_deg:.2f} | "
            "{overlap_with_uniform20:.3f} | {temporal_span:.2f} |".format(method=f"`{method}`", **metrics)
        )
    lines.extend(["", "## Dataset-wise", ""])
    for method, dataset_payload in output["summary"]["by_method_dataset"].items():
        lines.extend([f"### `{method}`", "", "| 数据集 | Azimuth 覆盖 | Elevation 覆盖 | Mean pair angle | Overlap vs uniform |", "|---|---:|---:|---:|---:|"])
        for dataset, metrics in dataset_payload.items():
            lines.append(
                "| {dataset} | {azimuth_coverage_deg:.2f} | {elevation_coverage_deg:.2f} | "
                "{mean_pairwise_spherical_angle_deg:.2f} | {overlap_with_uniform20:.3f} |".format(
                    dataset=f"`{dataset}`", **metrics
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
