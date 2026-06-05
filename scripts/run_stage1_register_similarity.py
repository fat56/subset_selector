#!/usr/bin/env python3
"""Compute Stage 1 register-token similarity against FastGS metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from vggt_omega_selector.backbones.vggt_omega import VGGTOmegaIntegration


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
METHODS = [f"random_ratio_seed{seed:03d}" for seed in range(5)] + ["uniform_stride_ratio"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 1 VGGT register similarity analysis.")
    parser.add_argument(
        "--jobs-tsv",
        default="runs/0001_stage1_register_quality_gate/queues/random_uniform_images4_30k/jobs.tsv",
    )
    parser.add_argument(
        "--cache-root",
        default="caches/vggt_omega/0001_stage1_register_quality_gate/register_similarity_images512",
    )
    parser.add_argument(
        "--analysis-dir",
        default="runs/0001_stage1_register_quality_gate/register_similarity_images512",
    )
    parser.add_argument(
        "--doc-output-dir",
        default="docs/experiments/0001_stage1_register_quality_gate/register_similarity",
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
    args = parser.parse_args(argv)

    project_root = Path.cwd()
    jobs_tsv = project_root / args.jobs_tsv
    cache_root = project_root / args.cache_root
    analysis_dir = project_root / args.analysis_dir
    doc_output_dir = project_root / args.doc_output_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    run_jobs = load_run_jobs(jobs_tsv)
    if args.scene:
        requested = set(args.scene)
        run_jobs = [job for job in run_jobs if job["scene_id"] in requested]
    if not run_jobs:
        raise SystemExit("No jobs matched.")

    cache_jobs, records = build_cache_jobs(run_jobs, cache_root)
    cache_jobs_path = analysis_dir / "cache_jobs.json"
    cache_jobs_path.write_text(json.dumps(cache_jobs, indent=2) + "\n", encoding="utf-8")

    if not args.skip_cache:
        jobs_for_run = cache_jobs[: args.max_cache_jobs] if args.max_cache_jobs else cache_jobs
        run_vggt_batch_cache(
            cache_jobs_path=write_limited_jobs(analysis_dir, jobs_for_run),
            checkpoint=args.checkpoint,
            image_resolution=args.image_resolution,
            mode=args.mode,
            device=args.device,
            force=args.force_cache,
        )
    if args.cache_only:
        print(json.dumps({"cache_jobs": len(cache_jobs[: args.max_cache_jobs] if args.max_cache_jobs else cache_jobs)}))
        return 0

    subset_rows, scene_rows = analyze(records)
    write_csv(doc_output_dir / "subset_register_similarity.csv", subset_rows)
    write_csv(doc_output_dir / "scene_register_correlations.csv", scene_rows)
    write_csv(analysis_dir / "subset_register_similarity.csv", subset_rows)
    write_csv(analysis_dir / "scene_register_correlations.csv", scene_rows)
    (analysis_dir / "summary.json").write_text(
        json.dumps({"subset_rows": subset_rows, "scene_rows": scene_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"subset_rows": len(subset_rows), "scene_rows": len(scene_rows)}, indent=2))
    return 0


def load_run_jobs(jobs_tsv: Path) -> list[dict[str, Any]]:
    with jobs_tsv.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_cache_jobs(run_jobs: list[dict[str, Any]], cache_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in run_jobs:
        by_scene[job["scene_id"]].append(job)

    cache_jobs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for scene_id in sorted(by_scene):
        scene_jobs = sorted(by_scene[scene_id], key=lambda job: METHODS.index(job["method"]))
        scene_images_dir = infer_scene_images_dir(Path(scene_jobs[0]["source_dir"]))
        full_names = sorted_image_names(scene_images_dir)
        image_map = {Path(name).stem: scene_images_dir / name for name in full_names}

        reference_split = json.loads((Path(scene_jobs[0]["source_dir"]) / "stage1_split.json").read_text(encoding="utf-8"))
        test_stems = {Path(name).stem for name in reference_split["test_images"]}
        llffhold_test_stems = {Path(name).stem for index, name in enumerate(full_names) if index % 8 == 0}
        if test_stems != llffhold_test_stems:
            raise RuntimeError(f"{scene_id}: stage1_split test set does not match full-scene llffhold")

        full_train_names = [name for name in full_names if Path(name).stem not in test_stems]
        full_cache_dir = cache_root / scene_id / "full_train_non_test"
        full_image_list = write_image_list(full_cache_dir / "image_list.txt", names_to_paths(full_train_names, image_map))
        cache_jobs.append(
            {
                "id": f"{scene_id}/full_train_non_test",
                "scene_id": scene_id,
                "method": "full_train_non_test",
                "role": "reference",
                "image_count": len(full_train_names),
                "image_list": str(full_image_list),
                "output_dir": str(full_cache_dir),
            }
        )
        records.append(
            {
                "scene_id": scene_id,
                "method": "full_train_non_test",
                "role": "reference",
                "cache_dir": str(full_cache_dir),
                "image_count": len(full_train_names),
            }
        )

        for job in scene_jobs:
            run_dir = Path(job["run_dir"])
            source_dir = Path(job["source_dir"])
            model_path = Path(job["model_path"])
            method = job["method"]
            split = json.loads((source_dir / "stage1_split.json").read_text(encoding="utf-8"))
            train_names = split["train_images"]
            if {Path(name).stem for name in split["test_images"]} != test_stems:
                raise RuntimeError(f"{scene_id}/{method}: inconsistent test split")
            if not {Path(name).stem for name in train_names}.isdisjoint(test_stems):
                raise RuntimeError(f"{scene_id}/{method}: train/test overlap")

            cache_dir = cache_root / scene_id / method
            image_list = write_image_list(cache_dir / "image_list.txt", names_to_paths(train_names, image_map))
            cache_jobs.append(
                {
                    "id": f"{scene_id}/{method}",
                    "scene_id": scene_id,
                    "method": method,
                    "role": "subset",
                    "image_count": len(train_names),
                    "image_list": str(image_list),
                    "output_dir": str(cache_dir),
                }
            )
            records.append(
                {
                    "scene_id": scene_id,
                    "method": method,
                    "role": "subset",
                    "cache_dir": str(cache_dir),
                    "image_count": len(train_names),
                    "result_path": str(model_path / "results.json"),
                    "run_dir": str(run_dir),
                }
            )
    return cache_jobs, records


def infer_scene_images_dir(source_dir: Path) -> Path:
    source_images = source_dir / "images"
    for image_path in sorted(source_images.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            target = image_path.resolve()
            parts = target.parts
            if "images" not in parts:
                raise RuntimeError(f"Cannot infer raw images dir from {target}")
            index = len(parts) - 1 - list(reversed(parts)).index("images")
            return Path(*parts[: index + 1])
    raise RuntimeError(f"No source images found under {source_images}")


def sorted_image_names(images_dir: Path) -> list[str]:
    return sorted(
        [path.name for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda name: Path(name).stem,
    )


def names_to_paths(names: list[str], image_map: dict[str, Path]) -> list[Path]:
    paths = []
    for name in names:
        stem = Path(name).stem
        try:
            paths.append(image_map[stem])
        except KeyError as exc:
            raise RuntimeError(f"Missing raw image for {name}") from exc
    return paths


def write_image_list(path: Path, image_paths: list[Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(image_path.resolve()) for image_path in image_paths) + "\n", encoding="utf-8")
    return path


def write_limited_jobs(analysis_dir: Path, jobs: list[dict[str, Any]]) -> Path:
    path = analysis_dir / "cache_jobs_to_run.json"
    path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return path


def run_vggt_batch_cache(
    *,
    cache_jobs_path: Path,
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
        str(cache_jobs_path),
        "--image-resolution",
        str(image_resolution),
        "--mode",
        mode,
        "--device",
        device,
    ]
    if force:
        command.append("--force")
    env = integration.subprocess_env()
    completed = subprocess.run(command, cwd=integration.project_root, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def analyze(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    references: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["role"] == "reference":
            references[record["scene_id"]] = record
        else:
            by_scene[record["scene_id"]].append(record)

    subset_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene_id in sorted(by_scene):
        reference_embedding = load_embedding(Path(references[scene_id]["cache_dir"]))
        scene_subset_rows = []
        for record in sorted(by_scene[scene_id], key=lambda item: METHODS.index(item["method"])):
            result = json.loads(Path(record["result_path"]).read_text(encoding="utf-8"))["ours_30000"]
            embedding = load_embedding(Path(record["cache_dir"]))
            row = {
                "scene_id": scene_id,
                "method": short_method(record["method"]),
                "image_count": record["image_count"],
                "register_mean_cosine": round(cosine(reference_embedding, embedding), 6),
                "psnr": round(float(result["PSNR"]), 6),
                "ssim": round(float(result["SSIM"]), 6),
                "lpips": round(float(result["LPIPS"]), 6),
            }
            subset_rows.append(row)
            scene_subset_rows.append(row)

        for metric in ("psnr", "ssim", "lpips"):
            values_x = [float(row["register_mean_cosine"]) for row in scene_subset_rows]
            values_y = [float(row[metric]) for row in scene_subset_rows]
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "metric": metric,
                    "n": len(scene_subset_rows),
                    "spearman": round(spearman(values_x, values_y), 6),
                    "pearson": round(pearson(values_x, values_y), 6),
                    "best_cosine_method": max(scene_subset_rows, key=lambda row: row["register_mean_cosine"])["method"],
                    "best_metric_method": max(scene_subset_rows, key=lambda row: row[metric])["method"]
                    if metric != "lpips"
                    else min(scene_subset_rows, key=lambda row: row[metric])["method"],
                }
            )
    return subset_rows, scene_rows


def load_embedding(cache_dir: Path) -> list[float]:
    path = cache_dir / "register_mean_embedding.json"
    if not path.exists():
        raise RuntimeError(f"Missing embedding: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [float(value) for value in payload["embedding"]]


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return float("nan")
    return numerator / (left_norm * right_norm)


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
