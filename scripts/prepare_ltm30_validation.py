#!/usr/bin/env python3
"""Prepare a 30-scene pose/depth validation manifest from local LTM datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Candidate:
    scene_key: str
    dataset: str
    source_group: str
    scene_root: Path
    scene_format: str
    priority: int
    depth_kind: str
    pose_kind: str
    eligible_frames: list[dict[str, Any]]
    scene_meta: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan data/raw/ltm_datasets and create an LTM30 validation manifest "
            "with full/random20/uniform20 splits."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/ltm_datasets"),
        help="Project-relative or absolute root for ltm_datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/experiments/0002_ltm30_pose_depth_validation"),
        help="Output directory for manifest and summaries.",
    )
    parser.add_argument("--scene-count", type=int, default=30)
    parser.add_argument("--max-full-frames", type=int, default=200)
    parser.add_argument("--subset-ratio", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=20260607)
    parser.add_argument("--min-eligible-frames", type=int, default=10)
    parser.add_argument(
        "--wildrgbd-candidates-per-category",
        type=int,
        default=1,
        help="How many valid WildRGBD scenes to keep per object category.",
    )
    parser.add_argument(
        "--dl3dv-candidate-limit",
        type=int,
        default=30,
        help="Maximum DL3DV scenes to inspect after sorting by scene id.",
    )
    parser.add_argument(
        "--include-pose-only",
        action="store_true",
        help="Allow pose-only scenes, e.g. ScanNet color+pose without depth.",
    )
    parser.add_argument(
        "--no-validate-selected-metadata",
        action="store_true",
        help="Skip loading selected metadata files to confirm pose keys.",
    )
    return parser.parse_args()


def as_rel(path: Path) -> str:
    return path.as_posix()


def stable_scene_seed(base_seed: int, scene_key: str) -> int:
    payload = f"{base_seed}:{scene_key}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def numeric_stem_key(path_or_stem: str | Path) -> tuple[int, str]:
    stem = Path(path_or_stem).stem if isinstance(path_or_stem, Path) else Path(path_or_stem).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return (int(digits), stem)
    return (10**18, stem)


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

    # Rounding can very rarely duplicate an index. Fill gaps deterministically.
    fill = 0
    while len(indices) < sample_count:
        if fill not in seen:
            indices.append(fill)
            seen.add(fill)
        fill += 1

    return sorted(indices)


def subset_count(full_count: int, ratio: float) -> int:
    return max(1, int(round(full_count * ratio)))


def sample_full_frames(frames: list[dict[str, Any]], max_full_frames: int) -> list[dict[str, Any]]:
    ordered = sorted(frames, key=lambda f: (f.get("sort_index", 10**18), f["frame_id"]))
    indices = uniform_indices(len(ordered), min(max_full_frames, len(ordered)))
    sampled: list[dict[str, Any]] = []
    for full_index, original_index in enumerate(indices):
        frame = dict(ordered[original_index])
        frame["full_index"] = full_index
        sampled.append(frame)
    return sampled


def make_splits(
    scene_key: str,
    full_frames: list[dict[str, Any]],
    ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    ids = [frame["frame_id"] for frame in full_frames]
    k = subset_count(len(ids), ratio)
    uniform_ids = [ids[i] for i in uniform_indices(len(ids), k)]

    rng = random.Random(stable_scene_seed(seed, scene_key))
    random_indices = sorted(rng.sample(range(len(ids)), k))
    random_ids = [ids[i] for i in random_indices]

    return {
        "full": ids,
        "random20": random_ids,
        "uniform20": uniform_ids,
    }


def find_one_image(stem: str, folder: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = folder / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def wildrgbd_frame_records(scene_dir: Path) -> list[dict[str, Any]]:
    rgb_dir = scene_dir / "rgb"
    depth_dir = scene_dir / "depth"
    meta_dir = scene_dir / "metadata"
    if not (rgb_dir.is_dir() and depth_dir.is_dir() and meta_dir.is_dir()):
        return []

    rgb_by_stem = {
        p.stem: p
        for p in rgb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    depth_by_stem = {
        p.stem: p
        for p in depth_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    }
    meta_by_stem = {
        p.stem: p
        for p in meta_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".npz"
    }
    stems = sorted(
        set(rgb_by_stem) & set(depth_by_stem) & set(meta_by_stem),
        key=numeric_stem_key,
    )

    frames: list[dict[str, Any]] = []
    for stem in stems:
        sort_index = numeric_stem_key(stem)[0]
        frames.append(
            {
                "frame_id": stem,
                "sort_index": sort_index,
                "image_path": as_rel(rgb_by_stem[stem]),
                "depth_path": as_rel(depth_by_stem[stem]),
                "pose_path": as_rel(meta_by_stem[stem]),
                "intrinsics_path": as_rel(meta_by_stem[stem]),
            }
        )
    return frames


def scan_wildrgbd(
    data_root: Path,
    min_frames: int,
    candidates_per_category: int,
) -> list[Candidate]:
    root = data_root / "wildrgbd_harrison"
    if not root.is_dir():
        return []

    candidates: list[Candidate] = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p / "scenes").is_dir()):
        kept = 0
        for scene_dir in sorted((category_dir / "scenes").iterdir(), key=lambda p: p.name):
            if not scene_dir.is_dir():
                continue
            frames = wildrgbd_frame_records(scene_dir)
            if len(frames) < min_frames:
                continue
            scene_key = f"wildrgbd_harrison/{category_dir.name}/{scene_dir.name}"
            candidates.append(
                Candidate(
                    scene_key=scene_key,
                    dataset="wildrgbd_harrison",
                    source_group=f"wildrgbd_harrison:{category_dir.name}",
                    scene_root=scene_dir,
                    scene_format="wildrgbd_rgb_depth_npz_pose",
                    priority=0,
                    depth_kind="sensor_depth_png",
                    pose_kind="npz_camera_pose",
                    eligible_frames=frames,
                    scene_meta={"category": category_dir.name, "scene_id": scene_dir.name},
                )
            )
            kept += 1
            if kept >= candidates_per_category:
                break
    return candidates


def valid_matrix4x4(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(isinstance(row, list) and len(row) == 4 for row in value)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def dl3dv_frame_records(scene_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transforms_path = scene_dir / "transforms.json"
    image_dir = scene_dir / "images_8"
    depth_dir = scene_dir / "colmap_depth_480p" / "depth_maps"
    depth_manifest_path = scene_dir / "colmap_depth_480p" / "manifest.json"

    transforms = load_json(transforms_path)
    if not transforms or not image_dir.is_dir():
        return [], {}

    frame_by_name: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(transforms.get("frames", [])):
        name = Path(str(frame.get("file_path", ""))).name
        matrix = frame.get("transform_matrix")
        if name and valid_matrix4x4(matrix):
            frame_by_name[name] = {
                "source_frame_index": index,
                "transform_matrix": matrix,
                "colmap_im_id": frame.get("colmap_im_id"),
            }

    names: list[str] = []
    depth_manifest = load_json(depth_manifest_path)
    if depth_manifest and isinstance(depth_manifest.get("frames"), list):
        for frame in depth_manifest["frames"]:
            name = frame.get("name")
            if isinstance(name, str):
                names.append(name)
    else:
        names = sorted(frame_by_name, key=numeric_stem_key)

    frames: list[dict[str, Any]] = []
    for name in sorted(set(names), key=numeric_stem_key):
        pose_meta = frame_by_name.get(name)
        if not pose_meta:
            continue
        image_path = image_dir / name
        depth_path = depth_dir / f"{name}.photometric.bin"
        if not image_path.is_file() or not depth_path.is_file():
            continue
        frame_id = Path(name).stem
        sort_index = numeric_stem_key(frame_id)[0]
        frames.append(
            {
                "frame_id": frame_id,
                "sort_index": sort_index,
                "image_path": as_rel(image_path),
                "depth_path": as_rel(depth_path),
                "pose_path": as_rel(transforms_path),
                "pose_frame_index": pose_meta["source_frame_index"],
                "colmap_im_id": pose_meta["colmap_im_id"],
                "transform_matrix": pose_meta["transform_matrix"],
            }
        )

    intrinsics = {}
    if depth_manifest:
        intrinsics = depth_manifest.get("intrinsics_scaled", {}) or {}

    scene_meta = {
        "scene_id": scene_dir.name,
        "intrinsics_scaled": intrinsics,
        "pose_convention": "opengl",
        "depth_manifest": as_rel(depth_manifest_path) if depth_manifest_path.is_file() else None,
    }
    return frames, scene_meta


def scan_dl3dv(data_root: Path, min_frames: int, limit: int) -> list[Candidate]:
    root = data_root / "DL3DV-ALL-480P" / "10K_extracted"
    if not root.is_dir():
        return []

    candidates: list[Candidate] = []
    for scene_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)[:limit]:
        frames, scene_meta = dl3dv_frame_records(scene_dir)
        if len(frames) < min_frames:
            continue
        scene_key = f"DL3DV-ALL-480P/10K_extracted/{scene_dir.name}"
        candidates.append(
            Candidate(
                scene_key=scene_key,
                dataset="DL3DV-ALL-480P",
                source_group="DL3DV-ALL-480P",
                scene_root=scene_dir,
                scene_format="nerf_transforms_images8_colmap_depth",
                priority=1,
                depth_kind="colmap_photometric_bin",
                pose_kind="transforms_json_opengl_c2w",
                eligible_frames=frames,
                scene_meta=scene_meta,
            )
        )
    return candidates


def scannet_frame_records(scene_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    color_dir = scene_dir / "color"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"
    if not (color_dir.is_dir() and pose_dir.is_dir()):
        return [], {}

    color_by_stem = {
        p.stem: p
        for p in color_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    pose_by_stem = {
        p.stem: p
        for p in pose_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    }
    stems = sorted(set(color_by_stem) & set(pose_by_stem), key=numeric_stem_key)

    frames: list[dict[str, Any]] = []
    for stem in stems:
        sort_index = numeric_stem_key(stem)[0]
        frames.append(
            {
                "frame_id": stem,
                "sort_index": sort_index,
                "image_path": as_rel(color_by_stem[stem]),
                "pose_path": as_rel(pose_by_stem[stem]),
            }
        )

    scene_meta = {
        "scene_id": scene_dir.name,
        "intrinsic_color": as_rel(intrinsic_dir / "intrinsic_color.txt")
        if (intrinsic_dir / "intrinsic_color.txt").is_file()
        else None,
        "intrinsic_depth": as_rel(intrinsic_dir / "intrinsic_depth.txt")
        if (intrinsic_dir / "intrinsic_depth.txt").is_file()
        else None,
    }
    return frames, scene_meta


def scan_scannet(data_root: Path, min_frames: int) -> list[Candidate]:
    root = data_root / "yifei_scannetv2_hf" / "scannetv2" / "scans"
    if not root.is_dir():
        return []

    candidates: list[Candidate] = []
    for scene_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        frames, scene_meta = scannet_frame_records(scene_dir)
        if len(frames) < min_frames:
            continue
        scene_key = f"yifei_scannetv2_hf/scannetv2/scans/{scene_dir.name}"
        candidates.append(
            Candidate(
                scene_key=scene_key,
                dataset="yifei_scannetv2_hf",
                source_group="yifei_scannetv2_hf",
                scene_root=scene_dir,
                scene_format="scannet_color_pose_intrinsics",
                priority=2,
                depth_kind="none",
                pose_kind="txt_camera_pose",
                eligible_frames=frames,
                scene_meta=scene_meta,
            )
        )
    return candidates


def select_candidates(candidates: list[Candidate], scene_count: int) -> list[Candidate]:
    selected: list[Candidate] = []
    for priority in sorted(set(c.priority for c in candidates)):
        priority_candidates = [c for c in candidates if c.priority == priority]
        groups: dict[str, list[Candidate]] = {}
        for candidate in sorted(priority_candidates, key=lambda c: (c.source_group, c.scene_key)):
            groups.setdefault(candidate.source_group, []).append(candidate)

        group_names = sorted(groups)
        depth = 0
        while len(selected) < scene_count:
            added_this_round = False
            for group_name in group_names:
                group = groups[group_name]
                if depth < len(group):
                    selected.append(group[depth])
                    added_this_round = True
                    if len(selected) >= scene_count:
                        break
            if not added_this_round:
                break
            depth += 1
        if len(selected) >= scene_count:
            break
    return selected


def validate_wildrgbd_selected(scenes: list[dict[str, Any]]) -> None:
    import numpy as np

    for scene in scenes:
        if scene["format"] != "wildrgbd_rgb_depth_npz_pose":
            continue
        for frame in scene["frames"]:
            pose_path = Path(frame["pose_path"])
            with np.load(pose_path) as data:
                keys = set(data.keys())
                missing = {"camera_intrinsics", "camera_pose"} - keys
                if missing:
                    raise ValueError(f"{pose_path} is missing keys: {sorted(missing)}")
                if data["camera_pose"].shape != (4, 4):
                    raise ValueError(f"{pose_path} camera_pose shape is {data['camera_pose'].shape}")
                if data["camera_intrinsics"].shape != (3, 3):
                    raise ValueError(
                        f"{pose_path} camera_intrinsics shape is {data['camera_intrinsics'].shape}"
                    )


def build_manifest(args: argparse.Namespace, candidates: list[Candidate]) -> dict[str, Any]:
    selected = select_candidates(candidates, args.scene_count)
    if len(selected) < args.scene_count:
        raise RuntimeError(
            f"Only found {len(selected)} valid scenes, but --scene-count={args.scene_count}."
        )

    scenes: list[dict[str, Any]] = []
    for candidate in selected:
        full_frames = sample_full_frames(candidate.eligible_frames, args.max_full_frames)
        splits = make_splits(candidate.scene_key, full_frames, args.subset_ratio, args.random_seed)
        scenes.append(
            {
                "scene_key": candidate.scene_key,
                "dataset": candidate.dataset,
                "source_group": candidate.source_group,
                "scene_root": as_rel(candidate.scene_root),
                "format": candidate.scene_format,
                "has_pose": True,
                "has_depth": candidate.depth_kind != "none",
                "depth_kind": candidate.depth_kind,
                "pose_kind": candidate.pose_kind,
                "eligible_frame_count": len(candidate.eligible_frames),
                "full_count": len(full_frames),
                "subset_count": subset_count(len(full_frames), args.subset_ratio),
                "scene_meta": candidate.scene_meta,
                "frames": full_frames,
                "splits": splits,
            }
        )

    by_dataset: dict[str, int] = {}
    by_depth_kind: dict[str, int] = {}
    for scene in scenes:
        by_dataset[scene["dataset"]] = by_dataset.get(scene["dataset"], 0) + 1
        by_depth_kind[scene["depth_kind"]] = by_depth_kind.get(scene["depth_kind"], 0) + 1

    candidate_summary: dict[str, int] = {}
    for candidate in candidates:
        key = f"{candidate.dataset}:{candidate.depth_kind}"
        candidate_summary[key] = candidate_summary.get(key, 0) + 1

    return {
        "manifest_version": 1,
        "created_at": date.today().isoformat(),
        "source_root": as_rel(args.data_root),
        "selection_policy": {
            "scene_count": args.scene_count,
            "max_full_frames": args.max_full_frames,
            "subset_ratio": args.subset_ratio,
            "random_seed": args.random_seed,
            "full_sampling": "uniform downsample over eligible frames, capped by max_full_frames",
            "random_subset": "deterministic scene-keyed random sample from full frames",
            "uniform_subset": "uniform sample from full frames",
            "priority_order": [
                "WildRGBD RGB + sensor depth + npz camera pose",
                "DL3DV images_8 + transforms pose + COLMAP photometric depth",
                "optional ScanNet color + pose only when --include-pose-only is set",
            ],
        },
        "summary": {
            "selected_scene_count": len(scenes),
            "selected_by_dataset": by_dataset,
            "selected_by_depth_kind": by_depth_kind,
            "candidate_summary": candidate_summary,
            "total_full_frames": sum(scene["full_count"] for scene in scenes),
            "total_random20_frames": sum(len(scene["splits"]["random20"]) for scene in scenes),
            "total_uniform20_frames": sum(len(scene["splits"]["uniform20"]) for scene in scenes),
        },
        "scenes": scenes,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_scene_csv(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_key",
                "dataset",
                "source_group",
                "format",
                "depth_kind",
                "pose_kind",
                "eligible_frame_count",
                "full_count",
                "subset_count",
                "scene_root",
            ],
        )
        writer.writeheader()
        for scene in manifest["scenes"]:
            writer.writerow(
                {
                    "scene_key": scene["scene_key"],
                    "dataset": scene["dataset"],
                    "source_group": scene["source_group"],
                    "format": scene["format"],
                    "depth_kind": scene["depth_kind"],
                    "pose_kind": scene["pose_kind"],
                    "eligible_frame_count": scene["eligible_frame_count"],
                    "full_count": scene["full_count"],
                    "subset_count": scene["subset_count"],
                    "scene_root": scene["scene_root"],
                }
            )


def write_summary_md(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    policy = manifest["selection_policy"]
    lines = [
        "# LTM30 Pose/Depth Validation Manifest",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Source root: `{manifest['source_root']}`",
        f"- Selected scenes: {summary['selected_scene_count']}",
        f"- Full frames: {summary['total_full_frames']}",
        f"- Random20 frames: {summary['total_random20_frames']}",
        f"- Uniform20 frames: {summary['total_uniform20_frames']}",
        f"- Max full frames per scene: {policy['max_full_frames']}",
        f"- Subset ratio: {policy['subset_ratio']}",
        f"- Random seed: {policy['random_seed']}",
        "",
        "## Selected Scene Counts",
        "",
        "| Key | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["selected_by_dataset"].items()):
        lines.append(f"| dataset:{key} | {value} |")
    for key, value in sorted(summary["selected_by_depth_kind"].items()):
        lines.append(f"| depth:{key} | {value} |")

    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| Scene | Dataset | Depth | Eligible | Full | 20% |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for scene in manifest["scenes"]:
        lines.append(
            "| "
            f"`{scene['scene_key']}` | "
            f"{scene['dataset']} | "
            f"{scene['depth_kind']} | "
            f"{scene['eligible_frame_count']} | "
            f"{scene['full_count']} | "
            f"{scene['subset_count']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    candidates = []
    candidates.extend(
        scan_wildrgbd(
            args.data_root,
            min_frames=args.min_eligible_frames,
            candidates_per_category=args.wildrgbd_candidates_per_category,
        )
    )
    candidates.extend(
        scan_dl3dv(
            args.data_root,
            min_frames=args.min_eligible_frames,
            limit=args.dl3dv_candidate_limit,
        )
    )
    if args.include_pose_only:
        candidates.extend(scan_scannet(args.data_root, min_frames=args.min_eligible_frames))

    manifest = build_manifest(args, candidates)
    if not args.no_validate_selected_metadata:
        validate_wildrgbd_selected(manifest["scenes"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "manifest.json", manifest)
    write_json(args.out_dir / "summary.json", {"summary": manifest["summary"]})
    write_scene_csv(args.out_dir / "scenes.csv", manifest)
    write_summary_md(args.out_dir / "summary.md", manifest)

    print(
        f"Wrote {manifest['summary']['selected_scene_count']} scenes to "
        f"{args.out_dir / 'manifest.json'}"
    )
    print(
        f"Full/random20/uniform20 frames: "
        f"{manifest['summary']['total_full_frames']}/"
        f"{manifest['summary']['total_random20_frames']}/"
        f"{manifest['summary']['total_uniform20_frames']}"
    )


if __name__ == "__main__":
    main()
