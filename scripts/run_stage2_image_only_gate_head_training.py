#!/usr/bin/env python3
"""Train an explicit uniform-fallback gate for 0005 image-only selectors."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_stage2_image_only_selector_training import (  # noqa: E402
    ImageOnlyCandidateDataset,
    ImageOnlyTrainConfig,
    MemoryCandidateSetScorer,
    candidate_stats,
    collate_candidate_batch,
    load_examples,
    mean,
    method_family,
    write_dataset_summary,
)


@dataclass
class GateHeadConfig:
    labels_csv: Path
    cache_jobs_json: Path
    feature_cache: Path
    run_dir: Path
    train_devices: list[str]
    candidate_tag: str = "20"
    seed: int = 20260609
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    limit_scenes: int | None = None
    epochs: int = 80
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    memory_slots: int = 8
    dropout: float = 0.1
    advantage_weight: float = 1.0
    gate_weight: float = 0.5
    rank_weight: float = 0.0
    positive_margin: float = 0.2
    advantage_clip: float = 4.0
    min_target_gap: float = 0.02
    target_gap_scale: float = 1.0
    advantage_thresholds: str = "-0.20,0.00,0.05,0.10,0.20,0.30,0.50,0.80,1.00,1.50,2.00"
    gate_thresholds: str = "-3.00,-2.00,-1.00,-0.50,0.00,0.50,1.00,2.00,3.00"
    num_workers: int = 0
    eval_every_epochs: int = 1
    log_every_steps: int = 20


class MemoryCandidateGateHead(nn.Module):
    """Predict uniform-relative advantage and a binary deviation gate per candidate."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        memory_slots: int,
        dropout: float,
        max_frames: int,
    ) -> None:
        super().__init__()
        self.context = MemoryCandidateSetScorer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            memory_slots=memory_slots,
            dropout=dropout,
            max_frames=max_frames,
        )
        feature_dim = hidden_dim * 4 + 8
        self.advantage_head = make_head(feature_dim, hidden_dim, dropout)
        self.gate_head = make_head(feature_dim, hidden_dim, dropout)

    def forward(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, memory = self.context.contextualize(features, frame_mask)
        candidate_features = self.build_candidate_features(hidden, memory, frame_mask, candidate_masks)
        advantages = self.advantage_head(candidate_features).squeeze(-1)
        gate_logits = self.gate_head(candidate_features).squeeze(-1)
        valid = candidate_valid.to(device=advantages.device, dtype=torch.bool)
        return advantages.masked_fill(~valid, 0.0), gate_logits.masked_fill(~valid, 0.0)

    @staticmethod
    def build_candidate_features(
        hidden: torch.Tensor,
        memory: torch.Tensor,
        frame_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
    ) -> torch.Tensor:
        selected = candidate_masks.to(dtype=hidden.dtype, device=hidden.device)
        selected = selected * frame_mask.to(dtype=hidden.dtype, device=hidden.device)[:, None, :]
        denom = selected.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_selected = torch.bmm(selected, hidden) / denom
        second = torch.bmm(selected, hidden * hidden) / denom
        std_selected = torch.sqrt((second - mean_selected * mean_selected).clamp_min(0.0) + 1e-6)

        full = frame_mask.to(dtype=hidden.dtype, device=hidden.device)
        full_denom = full.sum(dim=-1, keepdim=True).clamp_min(1.0)
        full_mean = (hidden * full[:, :, None]).sum(dim=1) / full_denom
        full_mean = full_mean[:, None, :].expand_as(mean_selected)
        memory_mean = memory.mean(dim=1)[:, None, :].expand_as(mean_selected)
        stats = candidate_stats(selected, frame_mask)
        return torch.cat([mean_selected, std_selected, full_mean, memory_mean, stats], dim=-1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = GateHeadConfig(
        labels_csv=resolve(args.labels_csv),
        cache_jobs_json=resolve(args.cache_jobs_json),
        feature_cache=resolve(args.feature_cache),
        run_dir=resolve(args.run_dir),
        train_devices=[device.strip() for device in args.train_devices.split(",") if device.strip()],
        candidate_tag=args.candidate_tag,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        limit_scenes=args.limit_scenes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        memory_slots=args.memory_slots,
        dropout=args.dropout,
        advantage_weight=args.advantage_weight,
        gate_weight=args.gate_weight,
        rank_weight=args.rank_weight,
        positive_margin=args.positive_margin,
        advantage_clip=args.advantage_clip,
        min_target_gap=args.min_target_gap,
        target_gap_scale=args.target_gap_scale,
        advantage_thresholds=args.advantage_thresholds,
        gate_thresholds=args.gate_thresholds,
        num_workers=args.num_workers,
        eval_every_epochs=args.eval_every_epochs,
        log_every_steps=args.log_every_steps,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "train_config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    set_random_seeds(config.seed)
    selector_config = to_selector_config(config)
    examples = load_examples(selector_config)
    write_dataset_summary(examples, selector_config)
    train_gate_head(config, examples)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an explicit image-only uniform fallback gate.")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--cache-jobs-json", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--advantage-weight", type=float, default=1.0)
    parser.add_argument("--gate-weight", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--positive-margin", type=float, default=0.2)
    parser.add_argument("--advantage-clip", type=float, default=4.0)
    parser.add_argument("--min-target-gap", type=float, default=0.02)
    parser.add_argument("--target-gap-scale", type=float, default=1.0)
    parser.add_argument("--advantage-thresholds", default="-0.20,0.00,0.05,0.10,0.20,0.30,0.50,0.80,1.00,1.50,2.00")
    parser.add_argument("--gate-thresholds", default="-3.00,-2.00,-1.00,-0.50,0.00,0.50,1.00,2.00,3.00")
    parser.add_argument("--train-devices", default="cuda:0,cuda:1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-steps", type=int, default=20)
    return parser.parse_args(argv)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_selector_config(config: GateHeadConfig) -> ImageOnlyTrainConfig:
    return ImageOnlyTrainConfig(
        labels_csv=config.labels_csv,
        cache_jobs_json=config.cache_jobs_json,
        feature_cache=config.feature_cache,
        run_dir=config.run_dir,
        train_devices=config.train_devices,
        candidate_tag=config.candidate_tag,
        model_kind="memory_candidate_set",
        seed=config.seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        limit_scenes=config.limit_scenes,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        memory_slots=config.memory_slots,
        dropout=config.dropout,
        min_target_gap=config.min_target_gap,
        target_gap_scale=config.target_gap_scale,
        num_workers=config.num_workers,
        eval_every_epochs=config.eval_every_epochs,
        log_every_steps=config.log_every_steps,
    )


def make_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, 1),
    )


def train_gate_head(config: GateHeadConfig, examples: list[Any]) -> dict[str, Any]:
    train_loader, val_loader, test_loader = build_loaders(config, examples)
    sample = ImageOnlyCandidateDataset(examples, "train")[0]
    input_dim = int(sample["features"].shape[-1])
    max_frames = max(example.full_image_count for example in examples)
    model = MemoryCandidateGateHead(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        memory_slots=config.memory_slots,
        dropout=config.dropout,
        max_frames=max_frames,
    )
    device = torch.device(config.train_devices[0] if config.train_devices else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    if len(config.train_devices) > 1:
        model = nn.DataParallel(model, device_ids=[int(device_name.split(":")[-1]) for device_name in config.train_devices])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    step = 0
    best_val_improvement = -float("inf")
    best_scan: dict[str, Any] | None = None
    best_path = config.run_dir / "best_gate.pt"
    started = time.time()

    initial_records = collect_records(model, val_loader, device, config)
    initial_scan = select_best_scan(scan_records({"val": initial_records}, config), "val")
    print(json.dumps({"event": "eval_initial", **compact_scan(initial_scan)}, ensure_ascii=False), flush=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            advantages, gate_logits = forward_model(model, batch, device)
            target_errors = batch["target_errors"].to(device, non_blocking=True)
            candidate_valid = batch["candidate_valid"].to(device, non_blocking=True)
            uniform_indices = batch["uniform_indices"].to(device, non_blocking=True)
            loss, stats = gate_head_loss(advantages, gate_logits, target_errors, candidate_valid, uniform_indices, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % config.log_every_steps == 0:
                elapsed = time.time() - started
                steps_per_sec = step / max(elapsed, 1e-6)
                eta = (total_steps - step) / max(steps_per_sec, 1e-6)
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch,
                            "step": step,
                            "total_steps": total_steps,
                            "loss": round(float(loss.item()), 6),
                            "advantage_loss": round(stats["advantage_loss"], 6),
                            "gate_loss": round(stats["gate_loss"], 6),
                            "rank_loss": round(stats["rank_loss"], 6),
                            "gate_positive_rate": round(stats["gate_positive_rate"], 6),
                            "steps_per_sec": round(steps_per_sec, 4),
                            "eta_sec": round(eta, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if epoch % config.eval_every_epochs == 0:
            val_records = collect_records(model, val_loader, device, config)
            scan = select_best_scan(scan_records({"val": val_records}, config), "val")
            print(json.dumps({"event": "eval", "epoch": epoch, **compact_scan(scan)}, ensure_ascii=False), flush=True)
            improvement = float(scan["val"]["uniform_minus_learned_error"])
            if improvement > best_val_improvement:
                best_val_improvement = improvement
                best_scan = scan
                save_checkpoint(best_path, model, config, epoch, step, scan)
            save_checkpoint(config.run_dir / "last.pt", model, config, epoch, step, scan)

    if best_path.is_file():
        load_checkpoint(best_path, model, device)
    final_records = {
        "train": collect_records(model, train_loader, device, config),
        "val": collect_records(model, val_loader, device, config),
        "test": collect_records(model, test_loader, device, config),
    }
    scans = scan_records(final_records, config)
    best_by_val = select_best_scan(scans, "val")
    best_by_test = select_best_scan(scans, "test")
    summary = {
        "epochs": config.epochs,
        "steps": step,
        "elapsed_sec": round(time.time() - started, 2),
        "best_checkpoint": str(best_path),
        "best_val_scan_during_training": best_scan,
        "best_by_val": best_by_val,
        "best_by_test_oracle_scan": best_by_test,
        "student_input_boundary": "image_only_no_vggt_tokens",
    }
    (config.run_dir / "gate_scan.json").write_text(json.dumps({"scans": scans, **summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (config.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **compact_summary(summary)}, ensure_ascii=False), flush=True)
    return summary


def build_loaders(config: GateHeadConfig, examples: list[Any]) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    loaders = []
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        loaders.append(
            DataLoader(
                ImageOnlyCandidateDataset(examples, split),
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers,
                collate_fn=collate_candidate_batch,
                pin_memory=True,
            )
        )
    return tuple(loaders)  # type: ignore[return-value]


def forward_model(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = batch["features"].to(device, non_blocking=True)
    frame_mask = batch["frame_mask"].to(device, non_blocking=True)
    candidate_masks = batch["candidate_masks"].to(device, non_blocking=True)
    candidate_valid = batch["candidate_valid"].to(device, non_blocking=True)
    return model(features, frame_mask, candidate_masks, candidate_valid)


def gate_head_loss(
    advantages: torch.Tensor,
    gate_logits: torch.Tensor,
    target_errors: torch.Tensor,
    candidate_valid: torch.Tensor,
    uniform_indices: torch.Tensor,
    config: GateHeadConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = candidate_valid.to(device=advantages.device, dtype=torch.bool)
    target_errors = target_errors.to(device=advantages.device)
    uniform_indices = uniform_indices.to(device=advantages.device, dtype=torch.long)
    target_advantage = compute_target_advantage(target_errors, uniform_indices, config.advantage_clip)
    positive = (target_advantage > config.positive_margin) & valid
    uniform_mask = torch.zeros_like(valid)
    uniform_mask.scatter_(1, uniform_indices[:, None], True)
    train_gate_mask = valid & ~uniform_mask

    advantage_loss = F.smooth_l1_loss(advantages[valid], target_advantage[valid])
    gate_targets = positive.to(dtype=gate_logits.dtype)
    if train_gate_mask.any():
        gate_loss = F.binary_cross_entropy_with_logits(gate_logits[train_gate_mask], gate_targets[train_gate_mask])
    else:
        gate_loss = advantages.new_zeros(())
    rank_loss = advantage_rank_loss(advantages, target_advantage, valid, config)
    loss = config.advantage_weight * advantage_loss + config.gate_weight * gate_loss + config.rank_weight * rank_loss
    return loss, {
        "advantage_loss": float(advantage_loss.detach().item()),
        "gate_loss": float(gate_loss.detach().item()),
        "rank_loss": float(rank_loss.detach().item()),
        "gate_positive_rate": float(positive[train_gate_mask].float().mean().detach().item()) if train_gate_mask.any() else 0.0,
    }


def advantage_rank_loss(
    advantages: torch.Tensor,
    target_advantage: torch.Tensor,
    valid: torch.Tensor,
    config: GateHeadConfig,
) -> torch.Tensor:
    if config.rank_weight <= 0.0:
        return advantages.new_zeros(())
    adv_i = advantages.unsqueeze(2)
    adv_j = advantages.unsqueeze(1)
    target_i = target_advantage.unsqueeze(2)
    target_j = target_advantage.unsqueeze(1)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    better_i = (target_i > target_j + config.min_target_gap) & pair_valid
    if not better_i.any():
        return advantages.new_zeros(())
    gap_weight = ((target_i - target_j).clamp_min(0.0) / max(config.target_gap_scale, 1e-6)).clamp(0.25, 4.0)
    return (F.softplus(-(adv_i - adv_j))[better_i] * gap_weight[better_i]).mean()


def compute_target_advantage(
    target_errors: torch.Tensor,
    uniform_indices: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    uniform_errors = target_errors.gather(1, uniform_indices[:, None])
    return (uniform_errors - target_errors).clamp(-clip, clip)


def collect_records(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    config: GateHeadConfig,
) -> list[dict[str, Any]]:
    model.eval()
    records = []
    with torch.inference_mode():
        for batch in loader:
            advantages, gate_logits = forward_model(model, batch, device)
            advantages = advantages.detach().cpu()
            gate_logits = gate_logits.detach().cpu()
            target_errors = batch["target_errors"]
            candidate_valid = batch["candidate_valid"]
            for row in range(advantages.shape[0]):
                valid_indices = torch.nonzero(candidate_valid[row], as_tuple=False).flatten()
                methods = [batch["methods"][row][int(index.item())] for index in valid_indices]
                row_targets = target_errors[row, valid_indices].float()
                row_advantages = advantages[row, valid_indices].float()
                row_gate_logits = gate_logits[row, valid_indices].float()
                uniform_idx = methods.index(f"uniform{config.candidate_tag}")
                oracle_idx = int(torch.argmin(row_targets).item())
                non_uniform_indices = [index for index in range(len(methods)) if index != uniform_idx]
                records.append(
                    {
                        "scene_id": batch["scene_ids"][row],
                        "dataset": batch["datasets"][row],
                        "methods": methods,
                        "target_errors": [float(value) for value in row_targets.tolist()],
                        "advantages": [float(value) for value in row_advantages.tolist()],
                        "gate_logits": [float(value) for value in row_gate_logits.tolist()],
                        "uniform_idx": int(uniform_idx),
                        "uniform_error": float(row_targets[uniform_idx].item()),
                        "oracle_idx": oracle_idx,
                        "oracle_error": float(row_targets[oracle_idx].item()),
                        "best_non_uniform_by_advantage": int(max(non_uniform_indices, key=lambda index: float(row_advantages[index].item()))),
                        "best_non_uniform_by_gate": int(max(non_uniform_indices, key=lambda index: float(row_gate_logits[index].item()))),
                    }
                )
    return records


def scan_records(records_by_split: dict[str, list[dict[str, Any]]], config: GateHeadConfig) -> list[dict[str, Any]]:
    scans = []
    for mode, thresholds in (
        ("advantage", parse_thresholds(config.advantage_thresholds)),
        ("gate_logit", parse_thresholds(config.gate_thresholds)),
    ):
        for threshold in thresholds:
            scan = {"mode": mode, "threshold": threshold}
            for split, records in records_by_split.items():
                scan[split] = summarize_choices(records, mode, threshold, config.candidate_tag)
            scans.append(scan)
    return scans


def parse_thresholds(raw: str) -> list[float]:
    return sorted({float(value.strip()) for value in raw.split(",") if value.strip()})


def summarize_choices(
    records: list[dict[str, Any]],
    mode: str,
    threshold: float,
    candidate_tag: str,
) -> dict[str, Any]:
    chosen = []
    for record in records:
        if mode == "advantage":
            candidate_idx = int(record["best_non_uniform_by_advantage"])
            candidate_value = float(record["advantages"][candidate_idx])
        elif mode == "gate_logit":
            candidate_idx = int(record["best_non_uniform_by_gate"])
            candidate_value = float(record["gate_logits"][candidate_idx])
        else:
            raise ValueError(f"Unsupported mode={mode}")
        chosen_idx = candidate_idx if candidate_value >= threshold else int(record["uniform_idx"])
        chosen.append((record, chosen_idx))
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
        "deviation_rate": mean([float(value) for value in deviations]),
        "win_rate_vs_uniform": mean([float(value) for value in wins]),
        "oracle_top1_rate": mean([float(value) for value in oracle_top1]),
        "learned_method_counts": learned_method_counts,
        "oracle_method_counts": oracle_method_counts,
    }


def select_best_scan(scans: list[dict[str, Any]], split: str) -> dict[str, Any]:
    return max(
        scans,
        key=lambda scan: (
            scan[split]["uniform_minus_learned_error"],
            -scan[split]["deviation_rate"],
            scan[split]["win_rate_vs_uniform"],
        ),
    )


def compact_scan(scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": scan["mode"],
        "threshold": scan["threshold"],
        "val": scan.get("val", {}),
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    best_by_val = summary["best_by_val"]
    best_by_test = summary["best_by_test_oracle_scan"]
    return {
        "best_checkpoint": summary["best_checkpoint"],
        "best_by_val": {
            "mode": best_by_val["mode"],
            "threshold": best_by_val["threshold"],
            "val": best_by_val["val"],
            "test": best_by_val["test"],
        },
        "best_by_test_oracle_scan": {
            "mode": best_by_test["mode"],
            "threshold": best_by_test["threshold"],
            "val": best_by_test["val"],
            "test": best_by_test["test"],
        },
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: GateHeadConfig,
    epoch: int,
    step: int,
    metrics: dict[str, Any],
) -> None:
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "model_state": state,
            "config": asdict(config),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "student_input_boundary": "image_only_no_vggt_tokens",
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(payload["model_state"])


if __name__ == "__main__":
    raise SystemExit(main())
