#!/usr/bin/env python3
"""Prepare the first fixed-ratio selector manifest for experiment 0004."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class SceneCandidate:
    scene_key: str
    scene_id: str
    dataset: str
    source_group: str
    scene_root: Path
    frames: list[dict[str, Any]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build 0004 Stage 2 fixed-K selector main_v1 manifest.")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--ltm-root", type=Path, default=Path("data/raw/ltm_datasets"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/experiments/0004_stage2_fixed_k_selector_training"))
    parser.add_argument("--manifest-stem", default="main_v1")
    parser.add_argument("--random-seed", type=int, default=20260608)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--train-bridgedata", type=int, default=900)
    parser.add_argument("--train-nyuv2", type=int, default=449)
    parser.add_argument("--train-tartanair", type=int, default=63)
    parser.add_argument("--train-scannet", type=int, default=300)
    parser.add_argument("--train-bonn", type=int, default=16)
    parser.add_argument("--val-per-major", type=int, default=50)
    parser.add_argument("--test-per-major", type=int, default=50)
    parser.add_argument("--val-bonn", type=int, default=5)
    parser.add_argument("--test-bonn", type=int, default=5)
    parser.add_argument("--limit-total", type=int, default=None, help="Optional quick cap after split assignment.")
    args = parser.parse_args(argv)

    processed_root = resolve(args.processed_root)
    ltm_root = resolve(args.ltm_root)
    out_dir = resolve(args.out_dir)

    pools = {
        "bridgedata_v2": scan_generic_rgb_sequences(processed_root / "bridgedata_v2", "bridgedata_v2", 2, args.min_frames),
        "nyuv2": scan_generic_rgb_sequences(processed_root / "nyuv2", "nyuv2", 2, args.min_frames),
        "tartanair": scan_generic_rgb_sequences(processed_root / "tartanair", "tartanair", 2, args.min_frames),
        "bonn": scan_generic_rgb_sequences(processed_root / "bonn", "bonn", 2, args.min_frames),
        "yifei_scannetv2_hf": scan_yifei_scannet(ltm_root / "yifei_scannetv2_hf", args.min_frames),
    }

    split_targets = {
        "bridgedata_v2": (args.train_bridgedata, args.val_per_major, args.test_per_major),
        "nyuv2": (args.train_nyuv2, args.val_per_major, args.test_per_major),
        "tartanair": (args.train_tartanair, args.val_per_major, args.test_per_major),
        "yifei_scannetv2_hf": (args.train_scannet, args.val_per_major, args.test_per_major),
        "bonn": (args.train_bonn, args.val_bonn, args.test_bonn),
    }

    scenes: list[dict[str, Any]] = []
    for dataset, candidates in pools.items():
        train_count, val_count, test_count = split_targets[dataset]
        selected = select_split(candidates, train_count, val_count, test_count, args.random_seed, dataset)
        for split, split_candidates in selected.items():
            for candidate in split_candidates:
                sampled_frames = sample_frames(candidate.frames, args.max_frames)
                scenes.append(
                    {
                        "scene_key": candidate.scene_key,
                        "scene_id": candidate.scene_id,
                        "dataset": candidate.dataset,
                        "source_group": candidate.source_group,
                        "split": split,
                        "scene_root": rel(candidate.scene_root),
                        "eligible_frame_count": len(candidate.frames),
                        "frame_count": len(sampled_frames),
                        "frames": sampled_frames,
                    }
                )

    if args.limit_total:
        scenes = sorted(scenes, key=lambda s: (s["split"], s["dataset"], s["scene_key"]))[: args.limit_total]

    manifest = {
        "manifest_version": 1,
        "experiment_id": "0004_stage2_fixed_k_selector_training",
        "name": args.manifest_stem,
        "created_at": date.today().isoformat(),
        "selection_policy": {
            "random_seed": args.random_seed,
            "max_frames": args.max_frames,
            "min_frames": args.min_frames,
            "ratio": 0.20,
            "scan_source_preference": "Use data/raw/ltm_datasets/yifei_scannetv2_hf for ScanNet.",
            "cache_policy": "cache-light selector_features.pt; no depth/depth_conf/full dense VGGT outputs",
            "split_targets": split_targets,
        },
        "summary": summarize(scenes, pools),
        "scenes": scenes,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{args.manifest_stem}_manifest.json", manifest)
    write_scene_csv(out_dir / f"{args.manifest_stem}_scenes.csv", scenes)
    write_summary(out_dir / f"{args.manifest_stem}_summary.md", manifest)
    print(json.dumps(manifest["summary"], indent=2))
    return 0


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    absolute = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        try:
            return absolute.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return absolute.resolve().as_posix()


def numeric_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else 10**18, path.stem)


def safe_id(key: str) -> str:
    return key.replace("/", "__")


def stable_seed(seed: int, text: str) -> int:
    return int(hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()[:16], 16)


def find_images(rgb_dir: Path) -> list[Path]:
    if not rgb_dir.is_dir():
        return []
    return sorted(
        [p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=numeric_key,
    )


def scan_generic_rgb_sequences(root: Path, dataset: str, sequence_depth: int, min_frames: int) -> list[SceneCandidate]:
    if not root.is_dir():
        return []
    candidates = []
    for rgb_dir in root.glob("/".join(["*"] * sequence_depth) + "/rgb"):
        images = find_images(rgb_dir)
        if len(images) < min_frames:
            continue
        scene_root = rgb_dir.parent
        scene_key = f"{dataset}/{scene_root.relative_to(root).as_posix()}"
        candidates.append(
            SceneCandidate(
                scene_key=scene_key,
                scene_id=safe_id(scene_key),
                dataset=dataset,
                source_group=f"{dataset}:{scene_root.relative_to(root).parts[0]}",
                scene_root=scene_root,
                frames=frames_from_images(images),
            )
        )
    return sorted(candidates, key=lambda c: c.scene_key)


def scan_yifei_scannet(root: Path, min_frames: int) -> list[SceneCandidate]:
    scans_root = root / "scannetv2" / "scans"
    if not scans_root.is_dir():
        return []
    candidates = []
    for scene_root in sorted([p for p in scans_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        images = find_images(scene_root / "color")
        if len(images) < min_frames:
            continue
        scene_key = f"yifei_scannetv2_hf/scannetv2/scans/{scene_root.name}"
        candidates.append(
            SceneCandidate(
                scene_key=scene_key,
                scene_id=safe_id(scene_key),
                dataset="yifei_scannetv2_hf",
                source_group="yifei_scannetv2_hf:scannetv2",
                scene_root=scene_root,
                frames=frames_from_images(images),
            )
        )
    return candidates


def frames_from_images(images: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "frame_id": image.stem,
            "sort_index": numeric_key(image)[0],
            "image_path": rel(image),
        }
        for image in images
    ]


def sample_frames(frames: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    ordered = sorted(frames, key=lambda f: (int(f.get("sort_index", 10**18)), f["frame_id"]))
    if len(ordered) <= max_frames:
        return [dict(frame, full_index=index) for index, frame in enumerate(ordered)]
    indices = []
    seen = set()
    for i in range(max_frames):
        idx = round(i * (len(ordered) - 1) / (max_frames - 1))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    fill = 0
    while len(indices) < max_frames:
        if fill not in seen:
            indices.append(fill)
            seen.add(fill)
        fill += 1
    return [dict(ordered[idx], full_index=out_idx) for out_idx, idx in enumerate(sorted(indices))]


def select_split(
    candidates: list[SceneCandidate],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    dataset: str,
) -> dict[str, list[SceneCandidate]]:
    needed = train_count + val_count + test_count
    if len(candidates) < needed:
        raise RuntimeError(f"{dataset}: need {needed} scenes, found {len(candidates)}")
    ordered = list(candidates)
    random.Random(stable_seed(seed, dataset)).shuffle(ordered)
    return {
        "train": sorted(ordered[:train_count], key=lambda c: c.scene_key),
        "val": sorted(ordered[train_count : train_count + val_count], key=lambda c: c.scene_key),
        "test": sorted(ordered[train_count + val_count : needed], key=lambda c: c.scene_key),
    }


def summarize(scenes: list[dict[str, Any]], pools: dict[str, list[SceneCandidate]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available_by_dataset": {key: len(value) for key, value in sorted(pools.items())},
        "selected_scene_count": len(scenes),
        "selected_by_split": {},
        "selected_by_dataset": {},
        "total_frames": sum(int(scene["frame_count"]) for scene in scenes),
    }
    for scene in scenes:
        out["selected_by_split"][scene["split"]] = out["selected_by_split"].get(scene["split"], 0) + 1
        out["selected_by_dataset"][scene["dataset"]] = out["selected_by_dataset"].get(scene["dataset"], 0) + 1
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_scene_csv(path: Path, scenes: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_id", "scene_key", "dataset", "split", "source_group", "eligible_frame_count", "frame_count", "scene_root"],
            lineterminator="\n",
        )
        writer.writeheader()
        for scene in scenes:
            writer.writerow({key: scene[key] for key in writer.fieldnames or []})


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# Stage 2 Fixed-K Selector Main V1 Manifest 摘要",
        "",
        f"- 创建日期: {manifest['created_at']}",
        f"- 入选 scenes: {summary['selected_scene_count']}",
        f"- 总 frames: {summary['total_frames']}",
        f"- Cache policy: {manifest['selection_policy']['cache_policy']}",
        f"- ScanNet source: {manifest['selection_policy']['scan_source_preference']}",
        "",
        "## 可用 Scene 统计",
        "",
        "| Dataset | Available | Selected |",
        "|---|---:|---:|",
    ]
    for dataset, available in sorted(summary["available_by_dataset"].items()):
        selected = summary["selected_by_dataset"].get(dataset, 0)
        lines.append(f"| {dataset} | {available} | {selected} |")
    lines.extend(["", "## Split", "", "| Split | Scenes |", "|---|---:|"])
    for split, count in sorted(summary["selected_by_split"].items()):
        lines.append(f"| {split} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
