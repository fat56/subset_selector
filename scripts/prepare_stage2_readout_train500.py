#!/usr/bin/env python3
"""Prepare the 500-scene readout training manifest for experiment 0003."""

from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Build Stage 2.0 readout train500 manifest.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ltm_datasets"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/experiments/0003_stage2_readout_calibration"),
    )
    parser.add_argument(
        "--ltm30-manifest",
        type=Path,
        default=Path("docs/experiments/0002_ltm30_pose_depth_validation/manifest.json"),
    )
    parser.add_argument("--wildrgbd-scenes", type=int, default=250)
    parser.add_argument("--dl3dv-scenes", type=int, default=250)
    parser.add_argument("--full-frames", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260607)
    parser.add_argument("--min-eligible-frames", type=int, default=16)
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
    dl3dv = prep.scan_dl3dv(data_root, args.min_eligible_frames, 1_000_000)

    selected_wildrgbd = select_deterministic(wildrgbd, args.wildrgbd_scenes, args.random_seed, "wildrgbd")
    selected_dl3dv = select_deterministic(dl3dv, args.dl3dv_scenes, args.random_seed, "dl3dv")
    selected = selected_wildrgbd + selected_dl3dv
    selected.sort(key=lambda c: (c.dataset, c.scene_key))

    scenes = []
    for candidate in selected:
        full_frames = prep.sample_full_frames(candidate.eligible_frames, args.full_frames)
        scenes.append(
            {
                "scene_key": candidate.scene_key,
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
                "frames": full_frames,
                "scene_meta": candidate.scene_meta,
            }
        )

    manifest = {
        "manifest_version": 1,
        "experiment_id": "0003_stage2_readout_calibration",
        "created_at": date.today().isoformat(),
        "source_root": prep.as_rel(data_root),
        "selection_policy": {
            "name": "train500_full16_wildrgbd_dl3dv_balanced",
            "random_seed": args.random_seed,
            "wildrgbd_scenes": args.wildrgbd_scenes,
            "dl3dv_scenes": args.dl3dv_scenes,
            "total_scenes": len(scenes),
            "full_frames_per_scene": args.full_frames,
            "min_eligible_frames": args.min_eligible_frames,
            "excluded_manifest": prep.as_rel(ltm30_manifest_path),
            "excluded_scene_count": len(excluded),
            "full_sampling": "uniform downsample over eligible frames",
            "training_subsets": "online mask sampling from full cached tokens",
            "training_subset_ratios": [0.25, 0.5, 0.75],
            "hard_native_labels": "not generated for train500 MVP; LTM30 hard subset metrics are validation-only",
        },
        "summary": summarize(scenes),
        "scenes": scenes,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "train500_manifest.json", manifest)
    write_scene_csv(out_dir / "train500_scenes.csv", scenes)
    write_summary_md(out_dir / "train500_summary.md", manifest)
    print(json.dumps(manifest["summary"], indent=2))
    return 0


def load_prepare_module() -> Any:
    spec = importlib.util.spec_from_file_location("prepare_ltm30_validation_for_readout", PREPARE_LTM30)
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


def summarize(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, int] = {}
    by_depth_kind: dict[str, int] = {}
    frame_counts = []
    eligible_counts = []
    for scene in scenes:
        by_dataset[scene["dataset"]] = by_dataset.get(scene["dataset"], 0) + 1
        by_depth_kind[scene["depth_kind"]] = by_depth_kind.get(scene["depth_kind"], 0) + 1
        frame_counts.append(int(scene["full_count"]))
        eligible_counts.append(int(scene["eligible_frame_count"]))
    return {
        "selected_scene_count": len(scenes),
        "selected_by_dataset": by_dataset,
        "selected_by_depth_kind": by_depth_kind,
        "total_full_frames": sum(frame_counts),
        "full_frames_min": min(frame_counts) if frame_counts else None,
        "full_frames_max": max(frame_counts) if frame_counts else None,
        "eligible_frames_min": min(eligible_counts) if eligible_counts else None,
        "eligible_frames_max": max(eligible_counts) if eligible_counts else None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_scene_csv(path: Path, scenes: list[dict[str, Any]]) -> None:
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
                "scene_root",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for scene in scenes:
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
                    "scene_root": scene["scene_root"],
                }
            )


def write_summary_md(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    policy = manifest["selection_policy"]
    lines = [
        "# Stage 2.0 Readout Train500 Manifest",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Source root: `{manifest['source_root']}`",
        f"- Selected scenes: {summary['selected_scene_count']}",
        f"- Full frames per scene: {policy['full_frames_per_scene']}",
        f"- Total full frames: {summary['total_full_frames']}",
        f"- Random seed: {policy['random_seed']}",
        f"- Excluded validation scenes: {policy['excluded_scene_count']}",
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
            "## Training Scope",
            "",
            "- This manifest caches only the full-view token set for each scene.",
            "- Training samples subset masks online from the cached full tokens.",
            "- LTM30 hard subset native metrics remain validation-only for this MVP.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
