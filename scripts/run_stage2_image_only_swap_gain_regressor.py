#!/usr/bin/env python3
"""Train an image-only regressor for single-step swap gains."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_stage2_image_only_selector_training import (  # noqa: E402
    ImageOnlyTrainConfig,
    MemoryCandidateSetScorer,
    SceneExample,
    load_examples,
    mean,
    method_family,
)


@dataclass
class SwapGainConfig:
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
    epochs: int = 120
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    memory_slots: int = 8
    dropout: float = 0.1
    regression_weight: float = 1.0
    sign_weight: float = 0.3
    rank_weight: float = 0.3
    gain_clip: float = 4.0
    min_target_gap: float = 0.02
    target_gap_scale: float = 1.0
    thresholds: str = "-1.00,-0.50,-0.20,-0.10,0.00,0.05,0.10,0.20,0.30,0.50,0.80,1.00,1.50,2.00,3.00"
    num_workers: int = 0
    eval_every_epochs: int = 1
    log_every_steps: int = 20


@dataclass(frozen=True)
class SwapPair:
    method: str
    add_index: int
    remove_index: int
    target_gain: float
    target_error: float


@dataclass(frozen=True)
class SwapSceneExample:
    scene_id: str
    scene_key: str
    dataset: str
    split: str
    feature_path: Path
    full_image_count: int
    uniform_error: float
    pairs: list[SwapPair]


class SwapGainDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: list[SwapSceneExample], split: str) -> None:
        self.examples = [example for example in examples if example.split == split]
        if not self.examples:
            raise ValueError(f"No examples for split={split}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        payload = torch.load(example.feature_path, map_location="cpu")
        if payload.get("feature_kind") != "image_only_no_vggt_tokens":
            raise ValueError(f"Feature cache is not image-only: {example.feature_path}")
        features = payload["frame_features"].float()
        image_stats = payload.get("image_stats")
        if image_stats is not None:
            features = torch.cat([features, image_stats.float()], dim=-1)
        if features.shape[0] != example.full_image_count:
            raise ValueError(
                f"Feature/frame count mismatch for {example.scene_id}: "
                f"{features.shape[0]} vs {example.full_image_count}"
            )
        return {
            "scene_id": example.scene_id,
            "scene_key": example.scene_key,
            "dataset": example.dataset,
            "features": features,
            "uniform_error": torch.tensor(example.uniform_error, dtype=torch.float32),
            "add_indices": torch.tensor([pair.add_index for pair in example.pairs], dtype=torch.long),
            "remove_indices": torch.tensor([pair.remove_index for pair in example.pairs], dtype=torch.long),
            "target_gains": torch.tensor([pair.target_gain for pair in example.pairs], dtype=torch.float32),
            "target_errors": torch.tensor([pair.target_error for pair in example.pairs], dtype=torch.float32),
            "methods": [pair.method for pair in example.pairs],
        }


class SwapGainRegressor(nn.Module):
    """Predict target_error improvement for replacing one uniform frame with one candidate frame."""

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
        pair_dim = hidden_dim * 5 + 6
        self.gain_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
        add_indices: torch.Tensor,
        remove_indices: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> torch.Tensor:
        hidden, memory = self.context.contextualize(features, frame_mask)
        added = gather_frames(hidden, add_indices)
        removed = gather_frames(hidden, remove_indices)
        memory_mean = memory.mean(dim=1)[:, None, :].expand_as(added)
        stats = pair_position_stats(add_indices, remove_indices, frame_mask).to(dtype=hidden.dtype, device=hidden.device)
        pair_features = torch.cat([added, removed, added - removed, (added - removed).abs(), memory_mean, stats], dim=-1)
        gains = self.gain_head(pair_features).squeeze(-1)
        return gains.masked_fill(~pair_valid.to(device=gains.device, dtype=torch.bool), 0.0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SwapGainConfig(
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
        regression_weight=args.regression_weight,
        sign_weight=args.sign_weight,
        rank_weight=args.rank_weight,
        gain_clip=args.gain_clip,
        min_target_gap=args.min_target_gap,
        target_gap_scale=args.target_gap_scale,
        thresholds=args.thresholds,
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
    examples = load_swap_examples(config)
    write_dataset_summary(config, examples)
    train_swap_gain(config, examples)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image-only single-swap gain regressor.")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--cache-jobs-json", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--sign-weight", type=float, default=0.3)
    parser.add_argument("--rank-weight", type=float, default=0.3)
    parser.add_argument("--gain-clip", type=float, default=4.0)
    parser.add_argument("--min-target-gap", type=float, default=0.02)
    parser.add_argument("--target-gap-scale", type=float, default=1.0)
    parser.add_argument("--thresholds", default="-1.00,-0.50,-0.20,-0.10,0.00,0.05,0.10,0.20,0.30,0.50,0.80,1.00,1.50,2.00,3.00")
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


def to_selector_config(config: SwapGainConfig) -> ImageOnlyTrainConfig:
    return ImageOnlyTrainConfig(
        labels_csv=config.labels_csv,
        cache_jobs_json=config.cache_jobs_json,
        feature_cache=config.feature_cache,
        run_dir=config.run_dir,
        train_devices=config.train_devices,
        candidate_tag=config.candidate_tag,
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


def load_swap_examples(config: SwapGainConfig) -> list[SwapSceneExample]:
    base_examples = load_examples(to_selector_config(config))
    examples = []
    for example in base_examples:
        uniform = next((candidate for candidate in example.candidates if candidate.method == f"uniform{config.candidate_tag}"), None)
        if uniform is None:
            continue
        uniform_mask = uniform.mask.bool()
        pairs = []
        for candidate in sorted(example.candidates, key=lambda item: item.method):
            if not candidate.method.startswith(f"swapgain{config.candidate_tag}_dino1_rank"):
                continue
            candidate_mask = candidate.mask.bool()
            added = candidate_mask & ~uniform_mask
            removed = uniform_mask & ~candidate_mask
            if int(added.sum().item()) != 1 or int(removed.sum().item()) != 1:
                continue
            pairs.append(
                SwapPair(
                    method=candidate.method,
                    add_index=int(torch.nonzero(added, as_tuple=False).flatten()[0].item()),
                    remove_index=int(torch.nonzero(removed, as_tuple=False).flatten()[0].item()),
                    target_gain=float(uniform.target_error - candidate.target_error),
                    target_error=float(candidate.target_error),
                )
            )
        if not pairs:
            continue
        examples.append(
            SwapSceneExample(
                scene_id=example.scene_id,
                scene_key=example.scene_key,
                dataset=example.dataset,
                split=example.split,
                feature_path=example.feature_path,
                full_image_count=example.full_image_count,
                uniform_error=float(uniform.target_error),
                pairs=pairs,
            )
        )
    if not examples:
        raise RuntimeError("No single-swap examples found.")
    return examples


def write_dataset_summary(config: SwapGainConfig, examples: list[SwapSceneExample]) -> None:
    split_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    gains = []
    best_gains = []
    pair_counts = []
    for example in examples:
        split_counts[example.split] = split_counts.get(example.split, 0) + 1
        dataset_counts[example.dataset] = dataset_counts.get(example.dataset, 0) + 1
        example_gains = [pair.target_gain for pair in example.pairs]
        gains.extend(example_gains)
        best_gains.append(max(example_gains))
        pair_counts.append(len(example.pairs))
    payload = {
        "total_scenes": len(examples),
        "split_counts": split_counts,
        "dataset_counts": dataset_counts,
        "total_pairs": len(gains),
        "pairs_per_scene": summarize_values([float(value) for value in pair_counts]),
        "target_gain_summary": summarize_values(gains),
        "best_swap_gain_summary": summarize_values(best_gains),
        "feature_cache": str(config.feature_cache),
        "student_input_boundary": "image_only_no_vggt_tokens",
    }
    (config.run_dir / "dataset_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "dataset_ready", **payload}, ensure_ascii=False), flush=True)


def summarize_values(values: list[float]) -> dict[str, float]:
    value_mean = mean(values)
    variance = mean([(value - value_mean) ** 2 for value in values])
    return {
        "count": float(len(values)),
        "mean": value_mean,
        "std": variance**0.5,
        "min": min(values),
        "max": max(values),
        "positive_fraction": sum(1 for value in values if value > 0.0) / max(len(values), 1),
        "negative_fraction": sum(1 for value in values if value < 0.0) / max(len(values), 1),
    }


def collate_swap_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(items)
    max_frames = max(item["features"].shape[0] for item in items)
    max_pairs = max(item["add_indices"].shape[0] for item in items)
    feature_dim = items[0]["features"].shape[-1]
    features = torch.zeros((batch_size, max_frames, feature_dim), dtype=torch.float32)
    frame_mask = torch.zeros((batch_size, max_frames), dtype=torch.bool)
    add_indices = torch.zeros((batch_size, max_pairs), dtype=torch.long)
    remove_indices = torch.zeros((batch_size, max_pairs), dtype=torch.long)
    pair_valid = torch.zeros((batch_size, max_pairs), dtype=torch.bool)
    target_gains = torch.zeros((batch_size, max_pairs), dtype=torch.float32)
    target_errors = torch.zeros((batch_size, max_pairs), dtype=torch.float32)
    uniform_errors = torch.zeros((batch_size,), dtype=torch.float32)
    methods = []
    for row, item in enumerate(items):
        frame_count = int(item["features"].shape[0])
        pair_count = int(item["add_indices"].shape[0])
        features[row, :frame_count] = item["features"]
        frame_mask[row, :frame_count] = True
        add_indices[row, :pair_count] = item["add_indices"]
        remove_indices[row, :pair_count] = item["remove_indices"]
        pair_valid[row, :pair_count] = True
        target_gains[row, :pair_count] = item["target_gains"]
        target_errors[row, :pair_count] = item["target_errors"]
        uniform_errors[row] = item["uniform_error"]
        methods.append(item["methods"])
    return {
        "features": features,
        "frame_mask": frame_mask,
        "add_indices": add_indices,
        "remove_indices": remove_indices,
        "pair_valid": pair_valid,
        "target_gains": target_gains,
        "target_errors": target_errors,
        "uniform_errors": uniform_errors,
        "scene_ids": [item["scene_id"] for item in items],
        "scene_keys": [item["scene_key"] for item in items],
        "datasets": [item["dataset"] for item in items],
        "methods": methods,
    }


def gather_frames(hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_indices = indices.to(device=hidden.device, dtype=torch.long)[:, :, None].expand(-1, -1, hidden.shape[-1])
    return hidden.gather(1, gather_indices)


def pair_position_stats(add_indices: torch.Tensor, remove_indices: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    device = frame_mask.device
    counts = frame_mask.sum(dim=1).float().clamp_min(1.0)
    denom = (counts - 1.0).clamp_min(1.0)
    add_pos = add_indices.to(device=device, dtype=torch.float32) / denom[:, None]
    remove_pos = remove_indices.to(device=device, dtype=torch.float32) / denom[:, None]
    delta = add_pos - remove_pos
    return torch.stack(
        [
            add_pos,
            remove_pos,
            delta,
            delta.abs(),
            (delta > 0).float(),
            counts[:, None].expand_as(add_pos) / 100.0,
        ],
        dim=-1,
    )


def train_swap_gain(config: SwapGainConfig, examples: list[SwapSceneExample]) -> dict[str, Any]:
    train_loader, val_loader, test_loader = build_loaders(config, examples)
    sample = SwapGainDataset(examples, "train")[0]
    input_dim = int(sample["features"].shape[-1])
    max_frames = max(example.full_image_count for example in examples)
    model = SwapGainRegressor(
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
    best_val = -float("inf")
    best_scan: dict[str, Any] | None = None
    best_path = config.run_dir / "best_swap_gain.pt"
    started = time.time()

    initial_records = collect_records(model, val_loader, device, config)
    initial_scan = select_best_scan(scan_records({"val": initial_records}, config), "val")
    print(json.dumps({"event": "eval_initial", **compact_scan(initial_scan)}, ensure_ascii=False), flush=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            predictions = forward_model(model, batch, device)
            target_gains = batch["target_gains"].to(device, non_blocking=True)
            pair_valid = batch["pair_valid"].to(device, non_blocking=True)
            loss, stats = swap_gain_loss(predictions, target_gains, pair_valid, config)
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
                            "regression_loss": round(stats["regression_loss"], 6),
                            "sign_loss": round(stats["sign_loss"], 6),
                            "rank_loss": round(stats["rank_loss"], 6),
                            "sign_accuracy": round(stats["sign_accuracy"], 6),
                            "pairwise_accuracy": round(stats["pairwise_accuracy"], 6),
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
            if improvement > best_val:
                best_val = improvement
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
        "best_by_test_oracle_threshold": best_by_test,
        "student_input_boundary": "image_only_no_vggt_tokens",
    }
    (config.run_dir / "swap_gain_scan.json").write_text(json.dumps({"scans": scans, **summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (config.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **compact_summary(summary)}, ensure_ascii=False), flush=True)
    return summary


def build_loaders(config: SwapGainConfig, examples: list[SwapSceneExample]) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    loaders = []
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        loaders.append(
            DataLoader(
                SwapGainDataset(examples, split),
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers,
                collate_fn=collate_swap_batch,
                pin_memory=True,
            )
        )
    return tuple(loaders)  # type: ignore[return-value]


def forward_model(model: nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return model(
        batch["features"].to(device, non_blocking=True),
        batch["frame_mask"].to(device, non_blocking=True),
        batch["add_indices"].to(device, non_blocking=True),
        batch["remove_indices"].to(device, non_blocking=True),
        batch["pair_valid"].to(device, non_blocking=True),
    )


def swap_gain_loss(
    predictions: torch.Tensor,
    target_gains: torch.Tensor,
    pair_valid: torch.Tensor,
    config: SwapGainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = pair_valid.to(device=predictions.device, dtype=torch.bool)
    targets = target_gains.to(device=predictions.device, dtype=predictions.dtype).clamp(-config.gain_clip, config.gain_clip)
    regression_loss = F.smooth_l1_loss(predictions[valid], targets[valid])

    sign_mask = valid & (targets.abs() > config.min_target_gap)
    if sign_mask.any():
        sign_targets = (targets[sign_mask] > 0).to(dtype=predictions.dtype)
        sign_loss = F.binary_cross_entropy_with_logits(predictions[sign_mask], sign_targets)
        sign_accuracy = ((predictions[sign_mask] > 0) == (targets[sign_mask] > 0)).float().mean()
    else:
        sign_loss = predictions.new_zeros(())
        sign_accuracy = predictions.new_zeros(())

    rank_loss, pairwise_accuracy = swap_rank_loss(predictions, targets, valid, config)
    loss = config.regression_weight * regression_loss + config.sign_weight * sign_loss + config.rank_weight * rank_loss
    return loss, {
        "regression_loss": float(regression_loss.detach().item()),
        "sign_loss": float(sign_loss.detach().item()),
        "rank_loss": float(rank_loss.detach().item()),
        "sign_accuracy": float(sign_accuracy.detach().item()),
        "pairwise_accuracy": float(pairwise_accuracy.detach().item()),
    }


def swap_rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    config: SwapGainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_i = predictions.unsqueeze(2)
    pred_j = predictions.unsqueeze(1)
    target_i = targets.unsqueeze(2)
    target_j = targets.unsqueeze(1)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    better_i = (target_i > target_j + config.min_target_gap) & pair_valid
    if not better_i.any():
        zero = predictions.new_zeros(())
        return zero, zero
    gap_weight = ((target_i - target_j).clamp_min(0.0) / max(config.target_gap_scale, 1e-6)).clamp(0.25, 4.0)
    pred_diff = pred_i - pred_j
    loss = (F.softplus(-pred_diff)[better_i] * gap_weight[better_i]).mean()
    accuracy = (pred_diff[better_i] > 0).float().mean()
    return loss, accuracy


def collect_records(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    config: SwapGainConfig,
) -> list[dict[str, Any]]:
    model.eval()
    records = []
    with torch.inference_mode():
        for batch in loader:
            predictions = forward_model(model, batch, device).detach().cpu()
            target_gains = batch["target_gains"]
            target_errors = batch["target_errors"]
            pair_valid = batch["pair_valid"]
            for row in range(predictions.shape[0]):
                valid_indices = torch.nonzero(pair_valid[row], as_tuple=False).flatten()
                row_predictions = predictions[row, valid_indices].float()
                row_gains = target_gains[row, valid_indices].float()
                row_errors = target_errors[row, valid_indices].float()
                methods = [batch["methods"][row][int(index.item())] for index in valid_indices]
                uniform_error = float(batch["uniform_errors"][row].item())
                records.append(
                    {
                        "scene_id": batch["scene_ids"][row],
                        "dataset": batch["datasets"][row],
                        "methods": methods,
                        "predicted_gains": [float(value) for value in row_predictions.tolist()],
                        "target_gains": [float(value) for value in row_gains.tolist()],
                        "target_errors": [float(value) for value in row_errors.tolist()],
                        "uniform_error": uniform_error,
                    }
                )
    return records


def scan_records(records_by_split: dict[str, list[dict[str, Any]]], config: SwapGainConfig) -> list[dict[str, Any]]:
    scans = []
    for threshold in parse_thresholds(config.thresholds):
        scan = {"threshold": threshold}
        for split, records in records_by_split.items():
            scan[split] = summarize_choices(records, threshold, config.candidate_tag, config)
        scans.append(scan)
    return scans


def parse_thresholds(raw: str) -> list[float]:
    return sorted({float(value.strip()) for value in raw.split(",") if value.strip()})


def summarize_choices(
    records: list[dict[str, Any]],
    threshold: float,
    candidate_tag: str,
    config: SwapGainConfig,
) -> dict[str, Any]:
    learned_errors = []
    uniform_errors = []
    oracle_errors = []
    deviations = []
    wins = []
    oracle_top1 = []
    best_swap_gains = []
    chosen_predicted_gains = []
    gain_abs_errors = []
    sign_correct = []
    pair_correct = 0
    pair_total = 0
    learned_method_counts: dict[str, int] = {}
    oracle_method_counts: dict[str, int] = {}
    for record in records:
        predictions = record["predicted_gains"]
        gains = record["target_gains"]
        errors = record["target_errors"]
        methods = record["methods"]
        uniform_error = float(record["uniform_error"])
        best_pred_idx = int(max(range(len(predictions)), key=lambda index: predictions[index]))
        best_swap_idx = int(max(range(len(gains)), key=lambda index: gains[index]))
        oracle_is_swap = errors[best_swap_idx] < uniform_error
        oracle_error = float(errors[best_swap_idx]) if oracle_is_swap else uniform_error
        oracle_method = method_family(methods[best_swap_idx], candidate_tag) if oracle_is_swap else f"uniform{candidate_tag}"

        choose_swap = float(predictions[best_pred_idx]) >= threshold
        chosen_error = float(errors[best_pred_idx]) if choose_swap else uniform_error
        chosen_method = method_family(methods[best_pred_idx], candidate_tag) if choose_swap else f"uniform{candidate_tag}"

        learned_errors.append(chosen_error)
        uniform_errors.append(uniform_error)
        oracle_errors.append(oracle_error)
        deviations.append(float(choose_swap))
        wins.append(float(chosen_error < uniform_error))
        oracle_top1.append(float((choose_swap and oracle_is_swap and best_pred_idx == best_swap_idx) or (not choose_swap and not oracle_is_swap)))
        best_swap_gains.append(float(gains[best_swap_idx]))
        chosen_predicted_gains.append(float(predictions[best_pred_idx]))
        learned_method_counts[chosen_method] = learned_method_counts.get(chosen_method, 0) + 1
        oracle_method_counts[oracle_method] = oracle_method_counts.get(oracle_method, 0) + 1

        for prediction, gain in zip(predictions, gains, strict=True):
            gain_abs_errors.append(abs(float(prediction) - float(gain)))
            if abs(float(gain)) > config.min_target_gap:
                sign_correct.append(float((float(prediction) > 0.0) == (float(gain) > 0.0)))
        for i in range(len(gains)):
            for j in range(len(gains)):
                if float(gains[i]) > float(gains[j]) + config.min_target_gap:
                    pair_total += 1
                    pair_correct += int(float(predictions[i]) > float(predictions[j]))

    learned_mean = mean(learned_errors)
    uniform_mean = mean(uniform_errors)
    oracle_mean = mean(oracle_errors)
    return {
        "scenes": float(len(records)),
        "learned_mean_error": learned_mean,
        "uniform20_mean_error": uniform_mean,
        "swap_oracle20_mean_error": oracle_mean,
        "uniform_minus_learned_error": uniform_mean - learned_mean,
        "uniform_regret": uniform_mean - oracle_mean,
        "learned_regret": learned_mean - oracle_mean,
        "deviation_rate": mean(deviations),
        "win_rate_vs_uniform": mean(wins),
        "oracle_top1_rate": mean(oracle_top1),
        "best_swap_gain_mean": mean(best_swap_gains),
        "chosen_predicted_gain_mean": mean(chosen_predicted_gains),
        "gain_mae": mean(gain_abs_errors),
        "sign_accuracy": mean(sign_correct),
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
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
        "threshold": scan["threshold"],
        "val": scan.get("val", {}),
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    best_by_val = summary["best_by_val"]
    best_by_test = summary["best_by_test_oracle_threshold"]
    return {
        "best_checkpoint": summary["best_checkpoint"],
        "best_by_val": {
            "threshold": best_by_val["threshold"],
            "val": best_by_val["val"],
            "test": best_by_val["test"],
        },
        "best_by_test_oracle_threshold": {
            "threshold": best_by_test["threshold"],
            "val": best_by_test["val"],
            "test": best_by_test["test"],
        },
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: SwapGainConfig,
    epoch: int,
    step: int,
    scan: dict[str, Any],
) -> None:
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "model_state": state,
            "config": asdict(config),
            "epoch": epoch,
            "step": step,
            "scan": scan,
            "student_input_boundary": "image_only_no_vggt_tokens",
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.load_state_dict(payload["model_state"])


if __name__ == "__main__":
    raise SystemExit(main())
