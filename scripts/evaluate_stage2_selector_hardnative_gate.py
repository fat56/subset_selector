#!/usr/bin/env python3
"""Evaluate uniform-fallback gates for hard-native candidate selectors."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_stage2_selector_hardnative_candidate_training import (  # noqa: E402
    HardNativeCandidateDataset,
    HardNativeTrainConfig,
    build_model,
    collate_candidate_batch,
    compute_candidate_scores,
    load_examples,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    run_dir: Path
    checkpoint: Path
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    ce_weight: float
    min_target_gap: float


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    specs = [
        ModelSpec(
            name="candidate_set_4l",
            run_dir=resolve(args.primary_run_dir),
            checkpoint=resolve(args.primary_checkpoint),
            hidden_dim=args.primary_hidden_dim,
            num_layers=args.primary_num_layers,
            num_heads=args.primary_num_heads,
            dropout=args.primary_dropout,
            ce_weight=args.primary_ce_weight,
            min_target_gap=args.primary_min_target_gap,
        ),
        ModelSpec(
            name="rankonly_2l",
            run_dir=resolve(args.secondary_run_dir),
            checkpoint=resolve(args.secondary_checkpoint),
            hidden_dim=args.secondary_hidden_dim,
            num_layers=args.secondary_num_layers,
            num_heads=args.secondary_num_heads,
            dropout=args.secondary_dropout,
            ce_weight=args.secondary_ce_weight,
            min_target_gap=args.secondary_min_target_gap,
        ),
    ]
    base_config = build_config(args, specs[0])
    examples = load_examples(base_config)
    records = collect_records(args, specs, examples, device)
    scans = scan_rules(records, specs)
    result = summarize_scans(scans, records, specs)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "hardnative_gate_eval_done",
                "out": str(out_path),
                "best_by_val_all": compact_rule(result["best_by_val_all"]),
                "best_by_val_positive_margin": compact_rule(result["best_by_val_positive_margin"]),
                "best_by_test_oracle_scan": compact_rule(result["best_by_test_oracle_scan"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hard-native selector gates against uniform20.")
    parser.add_argument(
        "--labels-csv",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv",
    )
    parser.add_argument(
        "--cache-jobs-json",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json",
    )
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out",
        default="runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector/ensemble_gate_scan.json",
    )

    parser.add_argument(
        "--primary-run-dir",
        default="runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector",
    )
    parser.add_argument(
        "--primary-checkpoint",
        default="runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector/best_uniform_improvement.pt",
    )
    parser.add_argument("--primary-hidden-dim", type=int, default=512)
    parser.add_argument("--primary-num-layers", type=int, default=4)
    parser.add_argument("--primary-num-heads", type=int, default=8)
    parser.add_argument("--primary-dropout", type=float, default=0.1)
    parser.add_argument("--primary-ce-weight", type=float, default=0.3)
    parser.add_argument("--primary-min-target-gap", type=float, default=0.02)

    parser.add_argument(
        "--secondary-run-dir",
        default="runs/0004_stage2_fixed_k_selector_training/main_v3_rankonly_small_candidate_selector_cuda0",
    )
    parser.add_argument(
        "--secondary-checkpoint",
        default=(
            "runs/0004_stage2_fixed_k_selector_training/"
            "main_v3_rankonly_small_candidate_selector_cuda0/best_uniform_improvement.pt"
        ),
    )
    parser.add_argument("--secondary-hidden-dim", type=int, default=256)
    parser.add_argument("--secondary-num-layers", type=int, default=2)
    parser.add_argument("--secondary-num-heads", type=int, default=4)
    parser.add_argument("--secondary-dropout", type=float, default=0.2)
    parser.add_argument("--secondary-ce-weight", type=float, default=0.0)
    parser.add_argument("--secondary-min-target-gap", type=float, default=0.05)
    return parser.parse_args(argv)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_config(args: argparse.Namespace, spec: ModelSpec) -> HardNativeTrainConfig:
    return HardNativeTrainConfig(
        labels_csv=resolve(args.labels_csv),
        cache_jobs_json=resolve(args.cache_jobs_json),
        run_dir=spec.run_dir,
        feature_cache_dir=spec.run_dir / "feature_cache",
        train_devices=[args.device] if args.device else ["cuda:0" if torch.cuda.is_available() else "cpu"],
        candidate_tag=args.candidate_tag,
        model_kind="candidate_set",
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        limit_scenes=None,
        epochs=1,
        batch_size=args.batch_size,
        lr=1e-4,
        weight_decay=1e-4,
        hidden_dim=spec.hidden_dim,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        dropout=spec.dropout,
        rank_weight=1.0,
        ce_weight=spec.ce_weight,
        min_target_gap=spec.min_target_gap,
        target_gap_scale=1.0,
        num_workers=0,
        eval_every_epochs=1,
        log_every_steps=20,
        rebuild_feature_cache=False,
    )


def collect_records(
    args: argparse.Namespace,
    specs: list[ModelSpec],
    examples: list[Any],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    loaders = {
        split: DataLoader(
            HardNativeCandidateDataset(examples, split),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_candidate_batch,
            pin_memory=device.type == "cuda",
        )
        for split in ("train", "val", "test")
    }
    train_sample = HardNativeCandidateDataset(examples, "train")[0]
    input_dim = int(train_sample["features"].shape[-1])
    max_frames = max(int(example.full_image_count) for example in examples)
    models = {}
    for spec in specs:
        config = build_config(args, spec)
        model = build_model(config, input_dim, max_frames).to(device)
        payload = torch.load(spec.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        model.eval()
        models[spec.name] = (model, config)

    records = {split: [] for split in loaders}
    with torch.inference_mode():
        for split, loader in loaders.items():
            for batch in loader:
                all_scores = {
                    name: compute_candidate_scores(model, batch, device, config.model_kind).detach().cpu()
                    for name, (model, config) in models.items()
                }
                for row in range(len(batch["scene_ids"])):
                    valid_indices = torch.nonzero(batch["candidate_valid"][row], as_tuple=False).flatten()
                    methods = [batch["methods"][row][int(index.item())] for index in valid_indices]
                    targets = batch["target_errors"][row, valid_indices].float()
                    uniform_idx = methods.index(f"uniform{args.candidate_tag}")
                    oracle_idx = int(torch.argmin(targets).item())
                    record = {
                        "scene_id": batch["scene_ids"][row],
                        "dataset": batch["datasets"][row],
                        "methods": methods,
                        "target_errors": [float(value) for value in targets.tolist()],
                        "uniform_idx": uniform_idx,
                        "uniform_error": float(targets[uniform_idx].item()),
                        "oracle_idx": oracle_idx,
                        "oracle_error": float(targets[oracle_idx].item()),
                        "models": {},
                    }
                    for name, scores_tensor in all_scores.items():
                        scores = scores_tensor[row, valid_indices].float()
                        non_uniform_indices = [index for index in range(len(scores)) if index != uniform_idx]
                        best_non_idx = max(non_uniform_indices, key=lambda index: float(scores[index].item()))
                        uniform_score = float(scores[uniform_idx].item())
                        best_non_score = float(scores[best_non_idx].item())
                        pred_idx = int(torch.argmax(scores).item())
                        record["models"][name] = {
                            "pred_idx": pred_idx,
                            "pred_method": methods[pred_idx],
                            "pred_error": float(targets[pred_idx].item()),
                            "margin_vs_uniform": best_non_score - uniform_score,
                            "best_non_idx_by_score": int(best_non_idx),
                        }
                    records[split].append(record)
    return records


def scan_rules(records: dict[str, list[dict[str, Any]]], specs: list[ModelSpec]) -> list[dict[str, Any]]:
    primary, secondary = specs[0].name, specs[1].name
    scans = []
    primary_thresholds = threshold_grid(records["val"], primary)
    secondary_thresholds = threshold_grid(records["val"], secondary)
    for primary_threshold in primary_thresholds:
        for secondary_threshold in secondary_thresholds:
            rule = {
                "rule": "agree_margin",
                f"t_{primary}": primary_threshold,
                f"t_{secondary}": secondary_threshold,
            }
            add_split_metrics(rule, records, lambda rec, r=rule: choose_agree(rec, r, primary, secondary))
            scans.append(rule)
    for model_name in (primary, secondary):
        for threshold in threshold_grid(records["val"], model_name):
            rule = {"rule": "single_margin", "model": model_name, "threshold": threshold}
            add_split_metrics(rule, records, lambda rec, r=rule: choose_single(rec, r["model"], r["threshold"]))
            scans.append(rule)
    return scans


def threshold_grid(records: list[dict[str, Any]], model_name: str) -> list[float]:
    anchors = [-1e9, 0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    margins = [record["models"][model_name]["margin_vs_uniform"] for record in records]
    return sorted(set(anchors + margins))


def add_split_metrics(
    rule: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    chooser: Any,
) -> None:
    for split in ("train", "val", "test"):
        rule[split] = summarize_choice([(record, chooser(record)) for record in records[split]])


def choose_agree(record: dict[str, Any], rule: dict[str, Any], primary: str, secondary: str) -> int:
    primary_state = record["models"][primary]
    secondary_state = record["models"][secondary]
    primary_idx = primary_state["best_non_idx_by_score"]
    secondary_idx = secondary_state["best_non_idx_by_score"]
    if (
        primary_idx == secondary_idx
        and primary_idx != record["uniform_idx"]
        and primary_state["margin_vs_uniform"] >= rule[f"t_{primary}"]
        and secondary_state["margin_vs_uniform"] >= rule[f"t_{secondary}"]
    ):
        return int(primary_idx)
    return int(record["uniform_idx"])


def choose_single(record: dict[str, Any], model_name: str, threshold: float) -> int:
    state = record["models"][model_name]
    candidate_idx = state["best_non_idx_by_score"]
    if candidate_idx != record["uniform_idx"] and state["margin_vs_uniform"] >= threshold:
        return int(candidate_idx)
    return int(record["uniform_idx"])


def summarize_choice(chosen: list[tuple[dict[str, Any], int]]) -> dict[str, float]:
    learned = [record["target_errors"][index] for record, index in chosen]
    uniform = [record["uniform_error"] for record, _ in chosen]
    oracle = [record["oracle_error"] for record, _ in chosen]
    deviations = [index != record["uniform_idx"] for record, index in chosen]
    wins = [record["target_errors"][index] < record["uniform_error"] for record, index in chosen]
    oracle_top1 = [index == record["oracle_idx"] for record, index in chosen]
    learned_mean = mean(learned)
    uniform_mean = mean(uniform)
    return {
        "scenes": float(len(chosen)),
        "learned_mean_error": learned_mean,
        "uniform20_mean_error": uniform_mean,
        "oracle20_mean_error": mean(oracle),
        "uniform_minus_learned_error": uniform_mean - learned_mean,
        "deviation_rate": mean([float(value) for value in deviations]),
        "win_rate_vs_uniform": mean([float(value) for value in wins]),
        "oracle_top1_rate": mean([float(value) for value in oracle_top1]),
    }


def summarize_scans(
    scans: list[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    specs: list[ModelSpec],
) -> dict[str, Any]:
    best_by_val_all = select_best_by_val(scans)
    positive_margin_scans = [scan for scan in scans if rule_min_threshold(scan) >= 0.0]
    best_by_val_positive = select_best_by_val(positive_margin_scans)
    best_by_test = max(scans, key=lambda scan: scan["test"]["uniform_minus_learned_error"])
    top_val = sorted(scans, key=lambda scan: scan["val"]["uniform_minus_learned_error"], reverse=True)[:10]
    return {
        "source_checkpoints": {spec.name: str(spec.checkpoint) for spec in specs},
        "splits": {split: len(items) for split, items in records.items()},
        "best_by_val_all": best_by_val_all,
        "best_by_val_all_deviations": deviations_for_rule(best_by_val_all, records, specs),
        "best_by_val_positive_margin": best_by_val_positive,
        "best_by_val_positive_margin_deviations": deviations_for_rule(best_by_val_positive, records, specs),
        "best_by_test_oracle_scan": best_by_test,
        "best_by_test_deviations": deviations_for_rule(best_by_test, records, specs),
        "top_val": top_val,
    }


def select_best_by_val(scans: list[dict[str, Any]]) -> dict[str, Any]:
    return max(scans, key=lambda scan: (scan["val"]["uniform_minus_learned_error"], -scan["val"]["deviation_rate"]))


def rule_min_threshold(rule: dict[str, Any]) -> float:
    if rule["rule"] == "single_margin":
        return float(rule["threshold"])
    thresholds = [float(value) for key, value in rule.items() if key.startswith("t_")]
    return min(thresholds)


def deviations_for_rule(
    rule: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    specs: list[ModelSpec],
) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [
            {
                "scene_id": record["scene_id"],
                "dataset": record["dataset"],
                "method": record["methods"][chosen_idx],
                "chosen_error": record["target_errors"][chosen_idx],
                "uniform_error": record["uniform_error"],
                "oracle_error": record["oracle_error"],
            }
            for record, chosen_idx in choices_for_rule(rule, records[split], specs)
            if chosen_idx != record["uniform_idx"]
        ]
        for split in ("val", "test")
    }


def choices_for_rule(
    rule: dict[str, Any],
    records: list[dict[str, Any]],
    specs: list[ModelSpec],
) -> list[tuple[dict[str, Any], int]]:
    primary, secondary = specs[0].name, specs[1].name
    choices = []
    for record in records:
        if rule["rule"] == "agree_margin":
            chosen_idx = choose_agree(record, rule, primary, secondary)
        else:
            chosen_idx = choose_single(record, rule["model"], rule["threshold"])
        choices.append((record, chosen_idx))
    return choices


def compact_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in rule.items() if key not in {"train", "val", "test"}}


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
