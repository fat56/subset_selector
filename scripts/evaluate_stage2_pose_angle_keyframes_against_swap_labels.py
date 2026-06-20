#!/usr/bin/env python3
"""Proxy-evaluate pose-angle keyframes against existing 0007 single-swap labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare pose-angle subsets with 0007 swap labels.")
    parser.add_argument("--subsets-json", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--cache-jobs-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--candidate-tag", default="20")
    args = parser.parse_args(argv)

    subsets_path = resolve(args.subsets_json)
    labels_path = resolve(args.labels_csv)
    jobs_path = resolve(args.cache_jobs_json)
    out_md = resolve(args.out_md)
    out_json = resolve(args.out_json) if args.out_json else out_md.with_suffix(".json")

    subsets = json.loads(subsets_path.read_text(encoding="utf-8"))
    labels = load_labels(labels_path)
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    job_indices = load_job_indices(jobs, subsets)

    scene_results = []
    for scene in subsets["scenes"]:
        scene_results.append(evaluate_scene(scene, labels.get(scene["scene_id"], {}), job_indices.get(scene["scene_id"], {}), args.candidate_tag))

    method_names = [method for method in subsets["methods"] if method != f"uniform{args.candidate_tag}"]
    summary = summarize(scene_results, method_names)
    payload = {
        "experiment_id": "0008_stage2_pose_angle_keyframing",
        "subsets_json": str(subsets_path),
        "labels_csv": str(labels_path),
        "cache_jobs_json": str(jobs_path),
        "summary": summary,
        "scenes": scene_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"event": "pose_angle_proxy_done", **summary}, ensure_ascii=False), flush=True)
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_labels(path: Path) -> dict[str, dict[str, float]]:
    labels: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels[row["scene_id"]][row["method"]] = float(row["target_error"])
    return dict(labels)


def load_job_indices(jobs: list[dict[str, Any]], subsets: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    path_maps = {}
    for scene in subsets["scenes"]:
        path_maps[scene["scene_id"]] = {frame["image_path"]: int(frame["index"]) for frame in scene["frames"]}
    output: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for job in jobs:
        scene_id = job["scene_id"]
        if scene_id not in path_maps or job.get("role") == "reference":
            continue
        image_list = Path(job["image_list"])
        if not image_list.is_file():
            continue
        indices = []
        for line in image_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line in path_maps[scene_id]:
                indices.append(path_maps[scene_id][line])
        output[scene_id][job["method"]] = sorted(indices)
    return {key: dict(value) for key, value in output.items()}


def evaluate_scene(
    scene: dict[str, Any],
    labels: dict[str, float],
    job_indices: dict[str, list[int]],
    candidate_tag: str,
) -> dict[str, Any]:
    uniform_method = f"uniform{candidate_tag}"
    uniform_indices = set(scene["methods"][uniform_method]["indices"])
    uniform_error = labels.get(uniform_method)

    swap_by_pair = {}
    positive_added = set()
    negative_added = set()
    best_gain = None
    best_added = None
    if uniform_error is not None:
        for method, indices in job_indices.items():
            if not method.startswith(f"swapgain{candidate_tag}_"):
                continue
            swap_set = set(indices)
            added = sorted(swap_set - uniform_indices)
            removed = sorted(uniform_indices - swap_set)
            if len(added) != 1 or len(removed) != 1 or method not in labels:
                continue
            gain = uniform_error - labels[method]
            swap_by_pair[(added[0], removed[0])] = {"method": method, "gain": gain}
            if gain > 0:
                positive_added.add(added[0])
            elif gain < 0:
                negative_added.add(added[0])
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_added = added[0]

    methods = {}
    for method, payload in scene["methods"].items():
        if method == uniform_method:
            continue
        selected = set(payload["indices"])
        added = sorted(selected - uniform_indices)
        removed = sorted(uniform_indices - selected)
        mapped = None
        gain = None
        if len(added) == 1 and len(removed) == 1:
            mapped = swap_by_pair.get((added[0], removed[0]))
            if mapped is not None:
                gain = mapped["gain"]
        methods[method] = {
            "added_count": len(added),
            "removed_count": len(removed),
            "overlap_with_uniform20": len(selected & uniform_indices) / max(len(uniform_indices), 1),
            "single_swap_diff": len(added) == 1 and len(removed) == 1,
            "mapped_single_swap": mapped is not None,
            "mapped_method": mapped["method"] if mapped is not None else None,
            "mapped_gain": gain,
            "mapped_gain_positive": bool(gain is not None and gain > 0.0),
            "added_positive_hit_rate": len(set(added) & positive_added) / max(len(added), 1),
            "added_negative_hit_rate": len(set(added) & negative_added) / max(len(added), 1),
            "added_best_hit": bool(best_added is not None and best_added in set(added)),
        }

    return {
        "scene_id": scene["scene_id"],
        "scene_key": scene["scene_key"],
        "dataset": scene["dataset"],
        "uniform_error": uniform_error,
        "swap_label_count": len(swap_by_pair),
        "positive_swap_added_count": len(positive_added),
        "best_swap_gain": best_gain,
        "methods": methods,
    }


def summarize(scene_results: list[dict[str, Any]], method_names: list[str]) -> dict[str, Any]:
    by_method = {}
    by_dataset: dict[str, dict[str, Any]] = {}
    for method in method_names:
        rows = [scene["methods"][method] for scene in scene_results if method in scene["methods"]]
        by_method[method] = summarize_rows(rows)
        by_dataset[method] = {}
        for dataset in sorted({scene["dataset"] for scene in scene_results}):
            dataset_rows = [scene["methods"][method] for scene in scene_results if scene["dataset"] == dataset and method in scene["methods"]]
            by_dataset[method][dataset] = summarize_rows(dataset_rows)
    return {
        "scene_count": len(scene_results),
        "dataset_counts": dataset_counts(scene_results),
        "by_method": by_method,
        "by_method_dataset": by_dataset,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    mapped = [row for row in rows if row["mapped_single_swap"]]
    gains = [float(row["mapped_gain"]) for row in mapped if row["mapped_gain"] is not None]
    return {
        "scene_count": float(len(rows)),
        "mean_added_count": mean([float(row["added_count"]) for row in rows]),
        "mean_overlap_with_uniform20": mean([float(row["overlap_with_uniform20"]) for row in rows]),
        "single_swap_diff_rate": mean([float(row["single_swap_diff"]) for row in rows]),
        "mapped_single_swap_rate": mean([float(row["mapped_single_swap"]) for row in rows]),
        "mapped_gain_mean": mean(gains),
        "mapped_gain_positive_rate": mean([float(row["mapped_gain_positive"]) for row in mapped]),
        "added_positive_hit_rate": mean([float(row["added_positive_hit_rate"]) for row in rows]),
        "added_negative_hit_rate": mean([float(row["added_negative_hit_rate"]) for row in rows]),
        "added_best_hit_rate": mean([float(row["added_best_hit"]) for row in rows]),
    }


def dataset_counts(scene_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scene in scene_results:
        counts[scene["dataset"]] = counts.get(scene["dataset"], 0) + 1
    return counts


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 0008 与 0007 Single-Swap Labels 的代理评估",
        "",
        f"- Subsets: `{payload['subsets_json']}`",
        f"- Labels: `{payload['labels_csv']}`",
        f"- 场景数: `{payload['summary']['scene_count']}`",
        f"- 数据集分布: `{json.dumps(payload['summary']['dataset_counts'], ensure_ascii=False)}`",
        "",
        "## 方法汇总",
        "",
        "| 方法 | Added 数 | Overlap vs uniform | Single-swap diff | 可映射率 | 映射 gain mean | 映射正 gain | Added 命中正 swap | Added 命中 best swap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in payload["summary"]["by_method"].items():
        lines.append(
            "| {method} | {mean_added_count:.2f} | {mean_overlap_with_uniform20:.3f} | {single_swap_diff_rate:.3f} | "
            "{mapped_single_swap_rate:.3f} | {mapped_gain_mean:.4f} | {mapped_gain_positive_rate:.3f} | "
            "{added_positive_hit_rate:.3f} | {added_best_hit_rate:.3f} |".format(method=f"`{method}`", **metrics)
        )
    lines.extend(["", "## Dataset-wise", ""])
    for method, dataset_payload in payload["summary"]["by_method_dataset"].items():
        lines.extend([f"### `{method}`", "", "| 数据集 | 可映射率 | 映射 gain mean | Added 命中正 swap | Added 命中 best swap |", "|---|---:|---:|---:|---:|"])
        for dataset, metrics in dataset_payload.items():
            lines.append(
                "| {dataset} | {mapped_single_swap_rate:.3f} | {mapped_gain_mean:.4f} | "
                "{added_positive_hit_rate:.3f} | {added_best_hit_rate:.3f} |".format(dataset=f"`{dataset}`", **metrics)
            )
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
