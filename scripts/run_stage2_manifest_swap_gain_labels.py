#!/usr/bin/env python3
"""Generate manifest-based single-swap VGGT-native labels for 0007."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_stage2_image_only_richer_candidate_labels import (  # noqa: E402
    build_training_labels,
    cache_ready,
    compute_native_metrics,
    records_from_jobs,
    run_cache_jobs_parallel,
    validate_cache_records,
    write_csv,
    write_label_summary,
)
from run_stage2_image_only_swap_gain_labels import (  # noqa: E402
    build_swap_candidates,
    parse_int_list,
    uniform_indices,
    write_gain_diagnostics,
    write_swap_summary,
)


@dataclass(frozen=True)
class ManifestScene:
    scene_id: str
    scene_key: str
    dataset: str
    full_images: list[Path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate 0007 manifest-based swap-gain labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--single-swaps", type=int, default=8)
    parser.add_argument("--multi-swaps", default="")
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

    manifest_path = resolve(args.manifest)
    feature_cache = resolve(args.feature_cache)
    run_dir = resolve(args.run_dir)
    cache_root = resolve(args.cache_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    scenes = load_manifest_scenes(manifest_path, feature_cache, args.limit_scenes)
    if not scenes:
        raise RuntimeError("No manifest scenes with matching feature cache were found.")

    jobs, records = build_manifest_swap_jobs(
        scenes=scenes,
        run_dir=run_dir,
        cache_root=cache_root,
        feature_cache=feature_cache,
        candidate_tag=args.candidate_tag,
        seed=args.seed,
        single_swaps=args.single_swaps,
        multi_swaps=parse_int_list(args.multi_swaps),
    )
    (run_dir / "augmented_cache_jobs.json").write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "augmented_cache_records.json").write_text(
        json.dumps({"records": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_swap_summary(run_dir / "swap_gain_candidate_summary.json", scenes, [record for record in records if record["role"] != "reference"])

    if not args.skip_cache and not args.labels_only:
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

    if args.cache_only:
        print(json.dumps({"event": "cache_only_done", "jobs": len(jobs), "records": len(records)}), flush=True)
        return 0

    validate_cache_records(records)
    metric_rows = compute_native_metrics(
        records=records,
        max_pixels_per_image=args.max_pixels_per_image,
        max_pointmap_points=args.max_pointmap_points,
        epsilon=args.epsilon,
    )
    write_csv(run_dir / "augmented_hardlabel_native_metrics.csv", metric_rows)
    label_rows = build_training_labels(metric_rows, records)
    write_csv(run_dir / "augmented_hardlabel_train_labels.csv", label_rows)
    write_label_summary(run_dir / "augmented_hardlabel_summary.json", metric_rows, label_rows)
    write_gain_diagnostics(run_dir / "swap_gain_diagnostics.json", label_rows, args.candidate_tag)

    print(
        json.dumps(
            {
                "event": "manifest_swap_gain_labels_done",
                "scenes": len(scenes),
                "jobs": len(jobs),
                "label_rows": len(label_rows),
                "labels_csv": str((run_dir / "augmented_hardlabel_train_labels.csv").resolve()),
                "cache_jobs_json": str((run_dir / "augmented_cache_jobs.json").resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_manifest_scenes(manifest_path: Path, feature_cache: Path, limit_scenes: int | None) -> list[ManifestScene]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = []
    for scene in manifest["scenes"]:
        scene_id = scene.get("scene_id") or scene["scene_key"].replace("/", "__")
        if not (feature_cache / f"{scene_id}.pt").is_file():
            continue
        scenes.append(
            ManifestScene(
                scene_id=scene_id,
                scene_key=scene["scene_key"],
                dataset=scene["dataset"],
                full_images=[resolve(frame["image_path"]) for frame in scene["frames"]],
            )
        )
    scenes = sorted(scenes, key=lambda scene: (scene.dataset, scene.scene_id))
    return scenes[:limit_scenes] if limit_scenes is not None else scenes


def build_manifest_swap_jobs(
    *,
    scenes: list[ManifestScene],
    run_dir: Path,
    cache_root: Path,
    feature_cache: Path,
    candidate_tag: str,
    seed: int,
    single_swaps: int,
    multi_swaps: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for scene in scenes:
        count = len(scene.full_images)
        k = max(1, int(round(count * (int(candidate_tag) / 100.0))))
        candidates = {f"uniform{candidate_tag}": uniform_indices(count, k)}
        candidates.update(
            build_swap_candidates(
                scene=scene,
                feature_cache=feature_cache,
                candidate_tag=candidate_tag,
                seed=seed,
                single_swaps=single_swaps,
                multi_swaps=multi_swaps,
            )
        )
        all_methods = {"full": list(range(count)), **candidates}
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
            records.append(records_from_jobs([job])[0])
    return jobs, records


if __name__ == "__main__":
    raise SystemExit(main())
