#!/usr/bin/env python3
"""Prepare the hard-label readout manifest for experiment 0003."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARE_LTM30 = PROJECT_ROOT / "scripts" / "prepare_ltm30_validation.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage 2.0 hard-label readout manifest.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ltm_datasets"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/experiments/0003_stage2_readout_calibration"),
    )
    parser.add_argument(
        "--manifest-stem",
        default="hardlabel100",
        help="Output stem for manifest, scenes CSV, and summary files.",
    )
    parser.add_argument(
        "--name",
        default="hardlabel100_full100_80",
        help="Manifest name recorded in the JSON payload.",
    )
    parser.add_argument(
        "--ltm30-manifest",
        type=Path,
        default=Path("docs/experiments/0002_ltm30_pose_depth_validation/manifest.json"),
    )
    parser.add_argument("--wildrgbd-scenes", type=int, default=50)
    parser.add_argument("--dl3dv-scenes", type=int, default=50)
    parser.add_argument("--wildrgbd-full-frames", type=int, default=100)
    parser.add_argument("--dl3dv-full-frames", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=20260607)
    parser.add_argument("--min-eligible-frames", type=int, default=80)
    parser.add_argument("--dl3dv-candidate-limit", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    prep = load_prepare_module()
    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    ltm30_manifest_path = resolve_path(args.ltm30_manifest)

    excluded = load_excluded_scene_keys(ltm30_manifest_path)
    wildrgbd = [
        candidate
        for candidate in prep.scan_wildrgbd(data_root, args.min_eligible_frames, 1_000_000)
        if candidate.scene_key not in excluded
    ]
    dl3dv = [
        candidate
        for candidate in prep.scan_dl3dv(data_root, args.min_eligible_frames, args.dl3dv_candidate_limit)
        if candidate.scene_key not in excluded
    ]

    selected = (
        select_deterministic(wildrgbd, args.wildrgbd_scenes, args.random_seed, f"wildrgbd_{args.manifest_stem}")
        + select_deterministic(dl3dv, args.dl3dv_scenes, args.random_seed, f"dl3dv_{args.manifest_stem}")
    )
    selected.sort(key=lambda c: (c.dataset, c.scene_key))

    scenes = []
    for candidate in selected:
        full_limit = args.wildrgbd_full_frames if candidate.dataset == "wildrgbd_harrison" else args.dl3dv_full_frames
        full_frames = prep.sample_full_frames(candidate.eligible_frames, full_limit)
        frame_by_id = {frame["frame_id"]: frame for frame in full_frames}
        splits = make_hard_splits(candidate.scene_key, list(frame_by_id), args.random_seed)
        scenes.append(
            {
                "scene_key": candidate.scene_key,
                "scene_id": safe_scene_id(candidate.scene_key),
                "dataset": candidate.dataset,
                "source_group": candidate.source_group,
                "scene_root": prep.as_rel(candidate.scene_root),
                "format": candidate.scene_format,
                "has_pose": True,
                "has_depth": candidate.depth_kind != "none",
                "depth_kind": candidate.depth_kind,
                "pose_kind": candidate.pose_kind,
                "eligible_frame_count": len(candidate.eligible_frames),
                "full_count": len(full_frames),
                "hard_subset_count": len(splits) - 1,
                "frames": full_frames,
                "splits": splits,
                "scene_meta": candidate.scene_meta,
            }
        )

    manifest = {
        "manifest_version": 1,
        "experiment_id": "0003_stage2_readout_calibration",
        "name": args.name,
        "created_at": date.today().isoformat(),
        "source_root": prep.as_rel(data_root),
        "selection_policy": {
            "name": f"{args.name}_wildrgbd_dl3dv_balanced",
            "random_seed": args.random_seed,
            "wildrgbd_scenes": args.wildrgbd_scenes,
            "dl3dv_scenes": args.dl3dv_scenes,
            "total_scenes": len(scenes),
            "wildrgbd_full_frames": args.wildrgbd_full_frames,
            "dl3dv_full_frames": args.dl3dv_full_frames,
            "min_eligible_frames": args.min_eligible_frames,
            "excluded_manifest": prep.as_rel(ltm30_manifest_path),
            "excluded_scene_count": len(excluded),
            "full_sampling": "uniform downsample over eligible frames",
            "subset_methods": [
                "random20_seed000-004",
                "random50_seed000-002",
                "uniform20",
                "uniform50",
                "contiguous20_seed000",
                "contiguous50_seed000",
            ],
            "hard_native_labels": "subset-vs-full VGGT-native depth/pose/point-map consistency",
        },
        "summary": summarize(scenes),
        "scenes": scenes,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{args.manifest_stem}_manifest.json", manifest)
    write_scene_csv(out_dir / f"{args.manifest_stem}_scenes.csv", scenes)
    write_summary_md(out_dir / f"{args.manifest_stem}_summary.md", manifest)
    print(json.dumps(manifest["summary"], indent=2))
    return 0


def load_prepare_module() -> Any:
    spec = importlib.util.spec_from_file_location("prepare_ltm30_validation_for_hardlabel", PREPARE_LTM30)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PREPARE_LTM30}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_excluded_scene_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(scene["scene_key"]) for scene in payload.get("scenes", [])}


def select_deterministic(candidates: list[Any], count: int, seed: int, salt: str) -> list[Any]:
    if len(candidates) < count:
        raise RuntimeError(f"Need {count} {salt} scenes, found {len(candidates)}.")
    ordered = sorted(candidates, key=lambda c: c.scene_key)
    rng = random.Random(f"{seed}:{salt}")
    rng.shuffle(ordered)
    return sorted(ordered[:count], key=lambda c: c.scene_key)


def make_hard_splits(scene_key: str, frame_ids: list[str], seed: int) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {"full": frame_ids}
    for seed_index in range(5):
        splits[f"random20_seed{seed_index:03d}"] = random_subset(
            frame_ids, 0.20, stable_seed(seed, scene_key, f"random20:{seed_index}")
        )
    for seed_index in range(3):
        splits[f"random50_seed{seed_index:03d}"] = random_subset(
            frame_ids, 0.50, stable_seed(seed, scene_key, f"random50:{seed_index}")
        )
    splits["uniform20"] = uniform_subset(frame_ids, 0.20)
    splits["uniform50"] = uniform_subset(frame_ids, 0.50)
    splits["contiguous20_seed000"] = contiguous_subset(frame_ids, 0.20, stable_seed(seed, scene_key, "contiguous20:0"))
    splits["contiguous50_seed000"] = contiguous_subset(frame_ids, 0.50, stable_seed(seed, scene_key, "contiguous50:0"))
    return splits


def stable_seed(seed: int, scene_key: str, salt: str) -> int:
    payload = f"{seed}:{scene_key}:{salt}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def subset_count(full_count: int, ratio: float) -> int:
    return max(1, int(round(full_count * ratio)))


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


def uniform_subset(frame_ids: list[str], ratio: float) -> list[str]:
    k = subset_count(len(frame_ids), ratio)
    return [frame_ids[index] for index in uniform_indices(len(frame_ids), k)]


def random_subset(frame_ids: list[str], ratio: float, seed: int) -> list[str]:
    k = subset_count(len(frame_ids), ratio)
    rng = random.Random(seed)
    return [frame_ids[index] for index in sorted(rng.sample(range(len(frame_ids)), k))]


def contiguous_subset(frame_ids: list[str], ratio: float, seed: int) -> list[str]:
    k = subset_count(len(frame_ids), ratio)
    max_start = max(len(frame_ids) - k, 0)
    start = random.Random(seed).randint(0, max_start) if max_start else 0
    return frame_ids[start : start + k]


def safe_scene_id(scene_key: str) -> str:
    return scene_key.replace("/", "__")


def summarize(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, int] = {}
    full_counts = []
    subset_jobs = 0
    subset_frames = 0
    for scene in scenes:
        by_dataset[scene["dataset"]] = by_dataset.get(scene["dataset"], 0) + 1
        full_counts.append(int(scene["full_count"]))
        for method, frame_ids in scene["splits"].items():
            if method == "full":
                continue
            subset_jobs += 1
            subset_frames += len(frame_ids)
    return {
        "selected_scene_count": len(scenes),
        "selected_by_dataset": by_dataset,
        "full_frames_min": min(full_counts) if full_counts else None,
        "full_frames_max": max(full_counts) if full_counts else None,
        "total_full_frames": sum(full_counts),
        "hard_subset_jobs": subset_jobs,
        "total_vggt_jobs": len(scenes) + subset_jobs,
        "total_subset_frames": subset_frames,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_scene_csv(path: Path, scenes: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "scene_key",
                "dataset",
                "source_group",
                "format",
                "depth_kind",
                "pose_kind",
                "eligible_frame_count",
                "full_count",
                "hard_subset_count",
                "scene_root",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for scene in scenes:
            writer.writerow(
                {
                    "scene_id": scene["scene_id"],
                    "scene_key": scene["scene_key"],
                    "dataset": scene["dataset"],
                    "source_group": scene["source_group"],
                    "format": scene["format"],
                    "depth_kind": scene["depth_kind"],
                    "pose_kind": scene["pose_kind"],
                    "eligible_frame_count": scene["eligible_frame_count"],
                    "full_count": scene["full_count"],
                    "hard_subset_count": scene["hard_subset_count"],
                    "scene_root": scene["scene_root"],
                }
            )


def write_summary_md(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    policy = manifest["selection_policy"]
    lines = [
        "# Stage 2.0 Hard-Label Readout Manifest 摘要",
        "",
        f"- 创建日期: {manifest['created_at']}",
        f"- 数据根目录: `{manifest['source_root']}`",
        f"- 入选 scenes: {summary['selected_scene_count']}",
        f"- WildRGBD scenes 数量: {policy['wildrgbd_scenes']}",
        f"- DL3DV scenes 数量: {policy['dl3dv_scenes']}",
        f"- Full frames: WildRGBD {policy['wildrgbd_full_frames']}, DL3DV {policy['dl3dv_full_frames']}",
        f"- Hard subset jobs 数量: {summary['hard_subset_jobs']}",
        f"- VGGT cache jobs 总数: {summary['total_vggt_jobs']}",
        f"- Random seed: {policy['random_seed']}（固定随机种子）",
        f"- 排除的 validation scenes: {policy['excluded_scene_count']}",
        "",
        "## 入选 Scene 统计",
        "",
        "| Dataset | 数量 |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["selected_by_dataset"].items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Subset 方法",
            "",
            "- `random20_seed000` ... `random20_seed004`",
            "- `random50_seed000` ... `random50_seed002`",
            "- `uniform20`, `uniform50`",
            "- `contiguous20_seed000`, `contiguous50_seed000`",
            "",
            "## Label 目标",
            "",
            "Hard labels 由每个 subset 的 VGGT-native depth/pose/point-map 输出与 full-view VGGT cache 中同一批图像的输出对比得到。",
            "训练目标是按 scene 做 z-score 后的 `pose_rotation_mean_deg`、`pointmap_rmse_norm` 和 `depth_log_rmse` 求和。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
