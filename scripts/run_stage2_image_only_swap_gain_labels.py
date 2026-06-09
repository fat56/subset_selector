#!/usr/bin/env python3
"""Generate true VGGT-native swap-gain labels around uniform20 for 0005."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_stage2_image_only_richer_candidate_labels import (  # noqa: E402
    build_training_labels,
    cache_ready,
    compute_native_metrics,
    load_feature_matrix,
    load_scene_sources,
    records_from_jobs,
    resolve,
    run_cache_jobs_parallel,
    stable_seed,
    uniform_indices,
    validate_cache_records,
    write_csv,
    write_label_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build small true swap-gain candidate labels for 0005.")
    parser.add_argument(
        "--base-labels-csv",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv",
    )
    parser.add_argument(
        "--base-cache-jobs-json",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json",
    )
    parser.add_argument(
        "--existing-metrics-csv",
        default="runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_hardlabel_native_metrics.csv",
    )
    parser.add_argument(
        "--existing-cache-jobs-json",
        default="runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_cache_jobs.json",
    )
    parser.add_argument("--feature-cache", default="caches/image_features/0005/hardlabel300_dinov2_vits14")
    parser.add_argument("--run-dir", default="runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2")
    parser.add_argument(
        "--cache-root",
        default="caches/vggt_omega/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_images512",
    )
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--limit-scenes", type=int, default=40)
    parser.add_argument("--single-swaps", type=int, default=4)
    parser.add_argument("--multi-swaps", default="2,4")
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
    feature_cache = resolve(args.feature_cache)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    existing_metrics = read_csv(resolve(args.existing_metrics_csv))
    existing_scene_ids = {row["scene_id"] for row in existing_metrics}
    scenes = load_scene_sources(
        base_labels_csv=resolve(args.base_labels_csv),
        base_cache_jobs_json=resolve(args.base_cache_jobs_json),
        candidate_tag=args.candidate_tag,
        limit_scenes=None,
        seed=args.seed,
    )
    scenes = [scene for scene in scenes if scene.scene_id in existing_scene_ids and (feature_cache / f"{scene.scene_id}.pt").is_file()]
    scenes = scenes[: args.limit_scenes] if args.limit_scenes is not None else scenes
    if not scenes:
        raise RuntimeError("No scenes available for swap-gain labels.")

    swap_jobs, swap_records = build_swap_gain_jobs(
        scenes=scenes,
        run_dir=run_dir,
        cache_root=cache_root,
        feature_cache=feature_cache,
        candidate_tag=args.candidate_tag,
        seed=args.seed,
        single_swaps=args.single_swaps,
        multi_swaps=parse_int_list(args.multi_swaps),
    )
    (run_dir / "swap_gain_cache_jobs.json").write_text(json.dumps(swap_jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "swap_gain_cache_records.json").write_text(
        json.dumps({"records": swap_records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_swap_summary(run_dir / "swap_gain_candidate_summary.json", scenes, swap_records)

    if not args.skip_cache and not args.labels_only:
        missing_jobs = [job for job in swap_jobs if args.force_cache or not cache_ready(Path(job["output_dir"]))]
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
            print(json.dumps({"event": "cache_skip", "reason": "all swap-gain jobs already cached", "jobs": len(swap_jobs)}), flush=True)

    if args.cache_only:
        print(json.dumps({"event": "cache_only_done", "jobs": len(swap_jobs), "records": len(swap_records)}), flush=True)
        return 0

    selected_scene_ids = {scene.scene_id for scene in scenes}
    existing_jobs = dedupe_jobs(
        [
            job
            for job in json.loads(resolve(args.existing_cache_jobs_json).read_text(encoding="utf-8"))
            if job["scene_id"] in selected_scene_ids
        ]
    )
    reference_jobs = [job for job in existing_jobs if job["method"] == "full"]
    metric_records = records_from_jobs(reference_jobs) + swap_records
    validate_cache_records(metric_records)
    swap_metrics = compute_native_metrics(
        records=metric_records,
        max_pixels_per_image=args.max_pixels_per_image,
        max_pointmap_points=args.max_pointmap_points,
        epsilon=args.epsilon,
    )
    write_csv(run_dir / "swap_gain_native_metrics.csv", swap_metrics)

    existing_metric_rows = [row for row in existing_metrics if row["scene_id"] in selected_scene_ids]
    augmented_metric_rows = sorted(existing_metric_rows + swap_metrics, key=lambda row: (row["scene_id"], row["method"]))
    write_csv(run_dir / "augmented_hardlabel_native_metrics.csv", augmented_metric_rows)

    augmented_jobs = dedupe_jobs(existing_jobs + swap_jobs)
    (run_dir / "augmented_cache_jobs.json").write_text(json.dumps(augmented_jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    augmented_records = records_from_jobs(augmented_jobs)
    augmented_labels = build_training_labels(augmented_metric_rows, augmented_records)
    write_csv(run_dir / "augmented_hardlabel_train_labels.csv", augmented_labels)
    write_label_summary(run_dir / "augmented_hardlabel_summary.json", augmented_metric_rows, augmented_labels)
    write_gain_diagnostics(run_dir / "swap_gain_diagnostics.json", augmented_labels, args.candidate_tag)

    print(
        json.dumps(
            {
                "event": "swap_gain_labels_done",
                "scenes": len(scenes),
                "swap_jobs": len(swap_jobs),
                "swap_metric_rows": len(swap_metrics),
                "augmented_label_rows": len(augmented_labels),
                "labels_csv": str((run_dir / "augmented_hardlabel_train_labels.csv").resolve()),
                "cache_jobs_json": str((run_dir / "augmented_cache_jobs.json").resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def build_swap_gain_jobs(
    *,
    scenes: list[Any],
    run_dir: Path,
    cache_root: Path,
    feature_cache: Path,
    candidate_tag: str,
    seed: int,
    single_swaps: int,
    multi_swaps: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = []
    records = []
    for scene in scenes:
        candidates = build_swap_candidates(
            scene=scene,
            feature_cache=feature_cache,
            candidate_tag=candidate_tag,
            seed=seed,
            single_swaps=single_swaps,
            multi_swaps=multi_swaps,
        )
        for method, indices in candidates.items():
            image_paths = [scene.full_images[index] for index in indices]
            image_list = run_dir / "image_lists" / scene.scene_id / f"{method}.txt"
            image_list.parent.mkdir(parents=True, exist_ok=True)
            image_list.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")
            output_dir = cache_root / scene.scene_id / method
            job = {
                "id": f"{scene.scene_id}/{method}",
                "scene_id": scene.scene_id,
                "scene_key": scene.scene_key,
                "dataset": scene.dataset,
                "method": method,
                "role": "subset",
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
                    "role": "subset",
                    "image_count": len(image_paths),
                    "cache_dir": str(output_dir.resolve()),
                    "token_path": str((output_dir / "camera_and_register_tokens.pt").resolve()),
                }
            )
    return jobs, records


def build_swap_candidates(
    *,
    scene: Any,
    feature_cache: Path,
    candidate_tag: str,
    seed: int,
    single_swaps: int,
    multi_swaps: list[int],
) -> dict[str, list[int]]:
    count = len(scene.full_images)
    k = max(1, int(round(count * (int(candidate_tag) / 100.0))))
    base = uniform_indices(count, k)
    base_set = set(base)
    features = load_feature_matrix(feature_cache / f"{scene.scene_id}.pt")
    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    ranked_additions = rank_additions_by_distance(normalized, base)
    candidates: dict[str, list[int]] = {}

    for rank, add_index in enumerate(ranked_additions[:single_swaps]):
        indices = apply_feature_swaps(base, normalized, [add_index])
        method = f"swapgain{candidate_tag}_dino1_rank{rank:03d}"
        candidates[method] = indices

    rng = random.Random(stable_seed(seed, scene.scene_id, "swapgain_multi"))
    for swap_count in multi_swaps:
        if swap_count <= 1 or swap_count >= k:
            continue
        offsets = [0, max(1, len(ranked_additions) // 5)]
        for variant, offset in enumerate(offsets):
            additions = diversified_additions(
                ranked_additions[offset:] + ranked_additions[:offset],
                normalized,
                base_set,
                swap_count,
                rng,
            )
            indices = apply_feature_swaps(base, normalized, additions)
            method = f"swapgain{candidate_tag}_dino{swap_count}_seed{variant:03d}"
            candidates[method] = indices
    return {method: indices for method, indices in candidates.items() if len(indices) == k and set(indices) != base_set}


def rank_additions_by_distance(normalized: torch.Tensor, base: list[int]) -> list[int]:
    count = int(normalized.shape[0])
    base_tensor = torch.tensor(base, dtype=torch.long)
    base_set = set(base)
    similarity = normalized @ normalized[base_tensor].T
    min_distance = 1.0 - similarity.max(dim=1).values
    ranked = torch.argsort(min_distance, descending=True).tolist()
    return [index for index in ranked if index not in base_set]


def diversified_additions(
    ranked: list[int],
    normalized: torch.Tensor,
    base_set: set[int],
    swap_count: int,
    rng: random.Random,
) -> list[int]:
    selected = []
    for index in ranked:
        if index in base_set or index in selected:
            continue
        if selected:
            sims = normalized[index] @ normalized[selected].T
            if float(sims.max().item()) > 0.92 and rng.random() < 0.75:
                continue
        selected.append(index)
        if len(selected) >= swap_count:
            break
    return selected


def apply_feature_swaps(base: list[int], normalized: torch.Tensor, additions: list[int]) -> list[int]:
    current = set(base)
    removable = set(base)
    for add_index in additions:
        if add_index in current or not removable:
            continue
        remove_index = nearest_index(add_index, sorted(removable), normalized)
        current.remove(remove_index)
        removable.remove(remove_index)
        current.add(add_index)
    return sorted(current)


def nearest_index(query: int, candidates: list[int], normalized: torch.Tensor) -> int:
    candidate_tensor = torch.tensor(candidates, dtype=torch.long)
    sims = normalized[candidate_tensor] @ normalized[query]
    return candidates[int(torch.argmax(sims).item())]


def dedupe_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for job in sorted(jobs, key=lambda item: (item["scene_id"], item["method"], item["output_dir"])):
        key = (job["scene_id"], job["method"])
        if key not in by_key or not cache_ready(Path(by_key[key]["output_dir"])):
            by_key[key] = job
    return sorted(by_key.values(), key=lambda item: (item["scene_id"], item["method"]))


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_swap_summary(path: Path, scenes: list[Any], records: list[dict[str, Any]]) -> None:
    by_method: dict[str, int] = defaultdict(int)
    by_dataset: dict[str, int] = defaultdict(int)
    for scene in scenes:
        by_dataset[scene.dataset] += 1
    for record in records:
        by_method[record["method"]] += 1
    payload = {
        "scenes": len(scenes),
        "subset_jobs": len(records),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_method": dict(sorted(by_method.items())),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_gain_diagnostics(path: Path, labels: list[dict[str, Any]], candidate_tag: str) -> None:
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        rows_by_scene[row["scene_id"]].append(row)
    swap_wins = 0
    swap_oracles = 0
    scenes_with_swap = 0
    uniform_minus_best_swap = []
    oracle_counts: dict[str, int] = defaultdict(int)
    for rows in rows_by_scene.values():
        uniform = next((row for row in rows if row["method"] == f"uniform{candidate_tag}"), None)
        swaps = [row for row in rows if row["method"].startswith(f"swapgain{candidate_tag}_")]
        if uniform is None or not swaps:
            continue
        scenes_with_swap += 1
        uniform_error = float(uniform["target_error"])
        best_swap = min(swaps, key=lambda row: float(row["target_error"]))
        best_swap_error = float(best_swap["target_error"])
        uniform_minus_best_swap.append(uniform_error - best_swap_error)
        swap_wins += int(best_swap_error < uniform_error)
        oracle = min(rows, key=lambda row: float(row["target_error"]))
        oracle_method = oracle["method"]
        if oracle_method.startswith(f"swapgain{candidate_tag}_"):
            swap_oracles += 1
            oracle_method = f"swapgain{candidate_tag}"
        elif oracle_method.startswith(f"uniform_jitter{candidate_tag}_"):
            oracle_method = f"uniform_jitter{candidate_tag}"
        elif oracle_method.startswith(f"random{candidate_tag}_"):
            oracle_method = f"random{candidate_tag}"
        elif oracle_method.startswith(f"contiguous{candidate_tag}_"):
            oracle_method = f"contiguous{candidate_tag}"
        oracle_counts[oracle_method] += 1
    payload = {
        "scenes_with_swap": scenes_with_swap,
        "swap_best_win_rate_vs_uniform": swap_wins / max(scenes_with_swap, 1),
        "swap_oracle_rate": swap_oracles / max(scenes_with_swap, 1),
        "uniform_minus_best_swap_mean": mean(uniform_minus_best_swap),
        "uniform_minus_best_swap_min": min(uniform_minus_best_swap) if uniform_minus_best_swap else None,
        "uniform_minus_best_swap_max": max(uniform_minus_best_swap) if uniform_minus_best_swap else None,
        "oracle_counts": dict(sorted(oracle_counts.items())),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
