#!/usr/bin/env python3
"""Evaluate uniform-fallback margin gates for 0005 image-only selectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_stage2_image_only_selector_training import (  # noqa: E402
    ImageOnlyCandidateDataset,
    ImageOnlyTrainConfig,
    build_model,
    collate_candidate_batch,
    compute_candidate_scores,
    load_examples,
    method_family,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint_paths = [resolve(path) for path in args.checkpoint]
    base_config = load_config_from_checkpoint(checkpoint_paths[0], args, run_dir)
    examples = load_examples(base_config)
    records_by_checkpoint = {}
    for checkpoint_path in checkpoint_paths:
        checkpoint_name = checkpoint_path.stem
        config = load_config_from_checkpoint(checkpoint_path, args, run_dir)
        records_by_checkpoint[checkpoint_name] = collect_records(config, examples, checkpoint_path, device)

    result = {
        "run_dir": str(run_dir),
        "out": str(out_path),
        "device": str(device),
        "checkpoint_paths": {path.stem: str(path) for path in checkpoint_paths},
        "splits": {split: len(next(iter(records_by_checkpoint.values()))[split]) for split in ("train", "val", "test")},
        "checkpoints": {},
    }
    for checkpoint_name, records in records_by_checkpoint.items():
        scans = scan_margins(records, args.candidate_tag, parse_margin_grid(args.margins))
        best_by_val = select_best_by_val(scans)
        best_by_test = max(scans, key=lambda scan: scan["test"]["uniform_minus_learned_error"])
        result["checkpoints"][checkpoint_name] = {
            "scans": scans,
            "top_val": sorted(scans, key=lambda scan: scan["val"]["uniform_minus_learned_error"], reverse=True)[:10],
            "best_by_val": best_by_val,
            "best_by_val_deviations": deviations_for_rule(best_by_val, records, args.candidate_tag),
            "best_by_test_oracle_scan": best_by_test,
            "best_by_test_deviations": deviations_for_rule(best_by_test, records, args.candidate_tag),
        }
    result["best_checkpoint_by_val"] = select_best_checkpoint(result, "val")
    result["best_checkpoint_by_test_oracle_scan"] = select_best_checkpoint(result, "test")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "image_only_gate_eval_done",
                "out": str(out_path),
                "best_checkpoint_by_val": compact_result(result["best_checkpoint_by_val"]),
                "best_checkpoint_by_test_oracle_scan": compact_result(result["best_checkpoint_by_test_oracle_scan"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image-only selector uniform fallback gates.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--cache-jobs-json", default=None)
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint path. Can be provided multiple times.",
    )
    parser.add_argument("--margins", default="0,0.02,0.05,0.10,0.20,0.30,0.50,0.80,1.00,1.50,2.00")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    run_dir = resolve(args.run_dir)
    if not args.checkpoint:
        args.checkpoint = [
            str(run_dir / "best_uniform_improvement.pt"),
            str(run_dir / "best_val_error.pt"),
            str(run_dir / "last.pt"),
        ]
    if args.out is None:
        args.out = str(run_dir / "uniform_gate_scan.json")
    return args


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config_from_checkpoint(path: Path, args: argparse.Namespace, run_dir: Path) -> ImageOnlyTrainConfig:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_config = dict(payload["config"])
    raw_config["labels_csv"] = resolve(args.labels_csv or raw_config["labels_csv"])
    raw_config["cache_jobs_json"] = resolve(args.cache_jobs_json or raw_config["cache_jobs_json"])
    raw_config["feature_cache"] = resolve(args.feature_cache or raw_config["feature_cache"])
    raw_config["run_dir"] = run_dir
    raw_config["candidate_tag"] = args.candidate_tag
    raw_config["batch_size"] = args.batch_size
    raw_config["train_devices"] = [args.device] if args.device else [raw_config["train_devices"][0]]
    raw_config["num_workers"] = 0
    if args.seed is not None:
        raw_config["seed"] = args.seed
    if args.train_fraction is not None:
        raw_config["train_fraction"] = args.train_fraction
    if args.val_fraction is not None:
        raw_config["val_fraction"] = args.val_fraction
    return ImageOnlyTrainConfig(**raw_config)


def collect_records(
    config: ImageOnlyTrainConfig,
    examples: list[Any],
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    loaders = {
        split: DataLoader(
            ImageOnlyCandidateDataset(examples, split),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_candidate_batch,
            pin_memory=device.type == "cuda",
        )
        for split in ("train", "val", "test")
    }
    sample = ImageOnlyCandidateDataset(examples, "train")[0]
    input_dim = int(sample["features"].shape[-1])
    max_frames = max(example.full_image_count for example in examples)
    model = build_model(config, input_dim=input_dim, max_frames=max_frames).to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    records = {split: [] for split in loaders}
    with torch.inference_mode():
        for split, loader in loaders.items():
            for batch in loader:
                scores = compute_candidate_scores(model, batch, device, config.model_kind).detach().cpu()
                target_errors = batch["target_errors"]
                candidate_valid = batch["candidate_valid"]
                for row in range(scores.shape[0]):
                    valid_indices = torch.nonzero(candidate_valid[row], as_tuple=False).flatten()
                    methods = [batch["methods"][row][int(index.item())] for index in valid_indices]
                    row_scores = scores[row, valid_indices].float()
                    row_targets = target_errors[row, valid_indices].float()
                    uniform_idx = methods.index(f"uniform{config.candidate_tag}")
                    non_uniform_indices = [index for index in range(len(methods)) if index != uniform_idx]
                    best_non_idx = max(non_uniform_indices, key=lambda index: float(row_scores[index].item()))
                    raw_idx = int(torch.argmax(row_scores).item())
                    oracle_idx = int(torch.argmin(row_targets).item())
                    records[split].append(
                        {
                            "scene_id": batch["scene_ids"][row],
                            "dataset": batch["datasets"][row],
                            "methods": methods,
                            "scores": [float(value) for value in row_scores.tolist()],
                            "target_errors": [float(value) for value in row_targets.tolist()],
                            "uniform_idx": int(uniform_idx),
                            "uniform_error": float(row_targets[uniform_idx].item()),
                            "oracle_idx": int(oracle_idx),
                            "oracle_error": float(row_targets[oracle_idx].item()),
                            "raw_idx": raw_idx,
                            "best_non_idx_by_score": int(best_non_idx),
                            "margin_vs_uniform": float(row_scores[best_non_idx].item() - row_scores[uniform_idx].item()),
                        }
                    )
    return records


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])


def parse_margin_grid(raw: str) -> list[float]:
    margins = [float(value.strip()) for value in raw.split(",") if value.strip()]
    return sorted(set(margins))


def scan_margins(
    records: dict[str, list[dict[str, Any]]],
    candidate_tag: str,
    margins: list[float],
) -> list[dict[str, Any]]:
    adaptive_margins = sorted(set(margins + [record["margin_vs_uniform"] for record in records["val"]]))
    scans = []
    raw_rule = {"rule": "raw_argmax", "margin": None}
    add_split_metrics(raw_rule, records, candidate_tag, choose_raw)
    scans.append(raw_rule)
    for margin in adaptive_margins:
        rule = {"rule": "uniform_margin", "margin": margin}
        add_split_metrics(rule, records, candidate_tag, lambda record, threshold=margin: choose_margin(record, threshold))
        scans.append(rule)
    return scans


def add_split_metrics(
    rule: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    candidate_tag: str,
    chooser: Any,
) -> None:
    for split in ("train", "val", "test"):
        rule[split] = summarize_choice([(record, chooser(record)) for record in records[split]], candidate_tag)


def choose_raw(record: dict[str, Any]) -> int:
    return int(record["raw_idx"])


def choose_margin(record: dict[str, Any], margin: float) -> int:
    if record["margin_vs_uniform"] >= margin:
        return int(record["best_non_idx_by_score"])
    return int(record["uniform_idx"])


def summarize_choice(chosen: list[tuple[dict[str, Any], int]], candidate_tag: str) -> dict[str, Any]:
    learned = [record["target_errors"][index] for record, index in chosen]
    uniform = [record["uniform_error"] for record, _index in chosen]
    oracle = [record["oracle_error"] for record, _index in chosen]
    deviations = [index != record["uniform_idx"] for record, index in chosen]
    wins = [record["target_errors"][index] < record["uniform_error"] for record, index in chosen]
    oracle_top1 = [index == record["oracle_idx"] for record, index in chosen]
    learned_method_counts: dict[str, int] = {}
    oracle_method_counts: dict[str, int] = {}
    for record, index in chosen:
        learned_method = method_family(record["methods"][index], candidate_tag)
        oracle_method = method_family(record["methods"][record["oracle_idx"]], candidate_tag)
        learned_method_counts[learned_method] = learned_method_counts.get(learned_method, 0) + 1
        oracle_method_counts[oracle_method] = oracle_method_counts.get(oracle_method, 0) + 1
    learned_mean = mean(learned)
    uniform_mean = mean(uniform)
    oracle_mean = mean(oracle)
    return {
        "scenes": float(len(chosen)),
        "learned_mean_error": learned_mean,
        "uniform20_mean_error": uniform_mean,
        "oracle20_mean_error": oracle_mean,
        "uniform_minus_learned_error": uniform_mean - learned_mean,
        "uniform_regret": uniform_mean - oracle_mean,
        "learned_regret": learned_mean - oracle_mean,
        "regret_reduction_vs_uniform": uniform_mean - learned_mean,
        "deviation_rate": mean([float(value) for value in deviations]),
        "win_rate_vs_uniform": mean([float(value) for value in wins]),
        "oracle_top1_rate": mean([float(value) for value in oracle_top1]),
        "learned_method_counts": learned_method_counts,
        "oracle_method_counts": oracle_method_counts,
    }


def select_best_by_val(scans: list[dict[str, Any]]) -> dict[str, Any]:
    return max(scans, key=lambda scan: (scan["val"]["uniform_minus_learned_error"], -scan["val"]["deviation_rate"]))


def select_best_checkpoint(result: dict[str, Any], split: str) -> dict[str, Any]:
    metric_key = f"best_by_{split}" if split == "val" else "best_by_test_oracle_scan"
    checkpoint_name, payload = max(
        result["checkpoints"].items(),
        key=lambda item: item[1][metric_key][split]["uniform_minus_learned_error"],
    )
    return {"checkpoint": checkpoint_name, **payload[metric_key]}


def deviations_for_rule(
    rule: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    candidate_tag: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [
            {
                "scene_id": record["scene_id"],
                "dataset": record["dataset"],
                "method": record["methods"][chosen_idx],
                "family": method_family(record["methods"][chosen_idx], candidate_tag),
                "chosen_error": record["target_errors"][chosen_idx],
                "uniform_error": record["uniform_error"],
                "oracle_error": record["oracle_error"],
                "margin_vs_uniform": record["margin_vs_uniform"],
            }
            for record, chosen_idx in choices_for_rule(rule, records[split])
            if chosen_idx != record["uniform_idx"]
        ]
        for split in ("val", "test")
    }


def choices_for_rule(rule: dict[str, Any], records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    choices = []
    for record in records:
        if rule["rule"] == "raw_argmax":
            chosen_idx = choose_raw(record)
        else:
            chosen_idx = choose_margin(record, float(rule["margin"]))
        choices.append((record, chosen_idx))
    return choices


def compact_result(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in rule.items()
        if key in {"checkpoint", "rule", "margin", "val", "test"}
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
