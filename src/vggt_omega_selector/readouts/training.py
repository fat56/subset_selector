"""Training utilities for Stage 2.0 readout calibration."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from vggt_omega_selector.readouts.models import CrossAttentionReadout, PooledReadout


PRIMARY_VAL_METRICS = ("pose_rotation_mean_deg", "pointmap_rmse_norm", "depth_log_rmse")


@dataclass(frozen=True)
class TrainConfig:
    train_index: Path
    run_dir: Path
    val_metrics_csv: Path
    val_cache_root: Path
    device: str = "cuda:0"
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    subset_ratios: tuple[float, ...] = (0.25, 0.5, 0.75)
    hidden_dim: int = 512
    output_dim: int = 256
    dropout: float = 0.1
    loss_pos_weight: float = 1.0
    loss_nce_weight: float = 1.0
    loss_rank_weight: float = 0.5
    nce_temperature: float = 0.07
    rank_margin: float = 0.1
    seed: int = 20260607
    eval_every_epochs: int = 1


@dataclass(frozen=True)
class HardLabelTrainConfig:
    labels_csv: Path
    run_dir: Path
    val_metrics_csv: Path
    val_cache_root: Path
    devices: tuple[str, ...] = ("cuda:0",)
    epochs: int = 40
    batch_size: int = 12
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    pairs_per_scene: int = 48
    method_filters: tuple[str, ...] = ()
    min_target_margin_fraction: float = 0.0
    hidden_dim: int = 512
    output_dim: int = 256
    dropout: float = 0.1
    loss_pos_weight: float = 0.25
    loss_nce_weight: float = 0.25
    loss_rank_weight: float = 1.0
    nce_temperature: float = 0.07
    rank_margin: float = 0.15
    seed: int = 20260607
    eval_every_epochs: int = 1


@dataclass(frozen=True)
class AttentionHardLabelTrainConfig:
    labels_csv: Path
    run_dir: Path
    val_metrics_csv: Path
    val_cache_root: Path
    devices: tuple[str, ...] = ("cuda:0",)
    epochs: int = 30
    batch_size: int = 16
    lr: float = 8e-5
    weight_decay: float = 1e-4
    num_workers: int = 4
    pairs_per_scene_metric: int = 24
    metrics: tuple[str, ...] = PRIMARY_VAL_METRICS
    method_filters: tuple[str, ...] = ()
    min_metric_margin_fraction: float = 0.0
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    rank_margin: float = 0.15
    full_rank_weight: float = 0.5
    embedding_rank_weight: float = 0.1
    embedding_pos_weight: float = 0.05
    nce_weight: float = 0.05
    nce_temperature: float = 0.07
    seed: int = 20260607
    eval_every_epochs: int = 1


class FullTokenSceneDataset(Dataset[dict[str, Any]]):
    def __init__(self, index_path: Path) -> None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.records = payload["records"]
        if not self.records:
            raise ValueError(f"No records in {index_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tensor = torch.load(record["token_path"], map_location="cpu")
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        return {
            "scene_id": record["scene_id"],
            "tokens": tensor.to(dtype=torch.float32),
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scene_id": [item["scene_id"] for item in items],
        "tokens": torch.stack([item["tokens"] for item in items], dim=0),
    }


class HardLabelPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        labels_csv: Path,
        pairs_per_scene: int,
        seed: int,
        method_filters: tuple[str, ...] = (),
        min_target_margin_fraction: float = 0.0,
    ) -> None:
        all_rows = read_csv(labels_csv)
        rows = filter_rows_by_method(all_rows, method_filters)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["scene_id"], []).append(row)
        pairs: list[dict[str, Any]] = []
        rng = random.Random(seed)
        pair_count_by_scene: Counter[str] = Counter()
        for scene_id, scene_rows in sorted(grouped.items()):
            ordered = sorted(scene_rows, key=lambda row: float(row["target_error"]))
            target_values = [float(row["target_error"]) for row in ordered]
            min_margin = metric_margin_threshold(target_values, min_target_margin_fraction)
            candidates = []
            for good_index, good in enumerate(ordered):
                for bad in ordered[good_index + 1 :]:
                    margin = float(bad["target_error"]) - float(good["target_error"])
                    if margin >= min_margin:
                        candidates.append((margin, good, bad))
            candidates.sort(key=lambda item: item[0], reverse=True)
            if len(candidates) > pairs_per_scene:
                head = candidates[: max(1, pairs_per_scene // 3)]
                tail = candidates[max(1, pairs_per_scene // 3) :]
                rng.shuffle(tail)
                selected = head + tail[: max(0, pairs_per_scene - len(head))]
            else:
                selected = candidates
            for margin, good, bad in selected:
                pairs.append(
                    {
                        "scene_id": scene_id,
                        "full_token_path": good["full_token_path"],
                        "good_token_path": good["subset_token_path"],
                        "bad_token_path": bad["subset_token_path"],
                        "good_method": good["method"],
                        "bad_method": bad["method"],
                        "good_target_error": float(good["target_error"]),
                        "bad_target_error": float(bad["target_error"]),
                        "target_margin": float(margin),
                    }
                )
                pair_count_by_scene[scene_id] += 1
        if not pairs:
            raise ValueError(f"No training pairs built from {labels_csv}")
        self.pairs = pairs
        self.summary = {
            "labels_csv": str(labels_csv),
            "total_label_rows": len(all_rows),
            "filtered_label_rows": len(rows),
            "total_scenes": len({row["scene_id"] for row in all_rows}),
            "filtered_scenes": len(grouped),
            "scenes_with_pairs": len(pair_count_by_scene),
            "pairs": len(pairs),
            "pairs_per_scene_requested": pairs_per_scene,
            "method_filters": list(method_filters),
            "min_target_margin_fraction": min_target_margin_fraction,
            "rows_by_method": dict(sorted(Counter(row["method"] for row in rows).items())),
            "pairs_by_scene_min": min(pair_count_by_scene.values()) if pair_count_by_scene else 0,
            "pairs_by_scene_max": max(pair_count_by_scene.values()) if pair_count_by_scene else 0,
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.pairs[index]
        return {
            **record,
            "full_tokens": load_token_tensor(Path(record["full_token_path"])),
            "good_tokens": load_token_tensor(Path(record["good_token_path"])),
            "bad_tokens": load_token_tensor(Path(record["bad_token_path"])),
        }


class MultiMetricHardLabelPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        labels_csv: Path,
        metrics: tuple[str, ...],
        pairs_per_scene_metric: int,
        seed: int,
        method_filters: tuple[str, ...] = (),
        min_metric_margin_fraction: float = 0.0,
    ) -> None:
        all_rows = read_csv(labels_csv)
        rows = filter_rows_by_method(all_rows, method_filters)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["scene_id"], []).append(row)
        pairs: list[dict[str, Any]] = []
        rng = random.Random(seed)
        pair_count_by_metric: Counter[str] = Counter()
        pair_count_by_scene_metric: Counter[str] = Counter()
        for scene_id, scene_rows in sorted(grouped.items()):
            for metric_index, metric in enumerate(metrics):
                ordered = sorted(scene_rows, key=lambda row: float(row[metric]))
                metric_values = [float(row[metric]) for row in ordered]
                min_margin = metric_margin_threshold(metric_values, min_metric_margin_fraction)
                candidates = []
                for good_index, good in enumerate(ordered):
                    for bad in ordered[good_index + 1 :]:
                        margin = float(bad[metric]) - float(good[metric])
                        if margin >= min_margin:
                            candidates.append((margin, good, bad))
                candidates.sort(key=lambda item: item[0], reverse=True)
                if len(candidates) > pairs_per_scene_metric:
                    head = candidates[: max(1, pairs_per_scene_metric // 3)]
                    tail = candidates[max(1, pairs_per_scene_metric // 3) :]
                    rng.shuffle(tail)
                    selected = head + tail[: max(0, pairs_per_scene_metric - len(head))]
                else:
                    selected = candidates
                for margin, good, bad in selected:
                    pairs.append(
                        {
                            "scene_id": scene_id,
                            "metric": metric,
                            "metric_index": metric_index,
                            "full_token_path": good["full_token_path"],
                            "good_token_path": good["subset_token_path"],
                            "bad_token_path": bad["subset_token_path"],
                            "good_method": good["method"],
                            "bad_method": bad["method"],
                            "metric_margin": float(margin),
                        }
                    )
                    pair_count_by_metric[metric] += 1
                    pair_count_by_scene_metric[f"{scene_id}/{metric}"] += 1
        if not pairs:
            raise ValueError(f"No multi-metric training pairs built from {labels_csv}")
        self.pairs = pairs
        self.summary = {
            "labels_csv": str(labels_csv),
            "total_label_rows": len(all_rows),
            "filtered_label_rows": len(rows),
            "total_scenes": len({row["scene_id"] for row in all_rows}),
            "filtered_scenes": len(grouped),
            "scene_metric_groups_with_pairs": len(pair_count_by_scene_metric),
            "pairs": len(pairs),
            "pairs_per_scene_metric_requested": pairs_per_scene_metric,
            "metrics": list(metrics),
            "method_filters": list(method_filters),
            "min_metric_margin_fraction": min_metric_margin_fraction,
            "rows_by_method": dict(sorted(Counter(row["method"] for row in rows).items())),
            "pairs_by_metric": dict(sorted(pair_count_by_metric.items())),
            "pairs_by_scene_metric_min": (
                min(pair_count_by_scene_metric.values()) if pair_count_by_scene_metric else 0
            ),
            "pairs_by_scene_metric_max": (
                max(pair_count_by_scene_metric.values()) if pair_count_by_scene_metric else 0
            ),
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.pairs[index]
        return {
            **record,
            "full_tokens": load_token_tensor(Path(record["full_token_path"])),
            "good_tokens": load_token_tensor(Path(record["good_token_path"])),
            "bad_tokens": load_token_tensor(Path(record["bad_token_path"])),
        }


def collate_multimetric_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    full_tokens, full_mask = pad_token_batch([item["full_tokens"] for item in items])
    good_tokens, good_mask = pad_token_batch([item["good_tokens"] for item in items])
    bad_tokens, bad_mask = pad_token_batch([item["bad_tokens"] for item in items])
    return {
        "scene_id": [item["scene_id"] for item in items],
        "metric": [item["metric"] for item in items],
        "good_method": [item["good_method"] for item in items],
        "bad_method": [item["bad_method"] for item in items],
        "metric_index": torch.as_tensor([item["metric_index"] for item in items], dtype=torch.long),
        "metric_margin": torch.as_tensor([item["metric_margin"] for item in items], dtype=torch.float32),
        "full_tokens": full_tokens,
        "full_mask": full_mask,
        "good_tokens": good_tokens,
        "good_mask": good_mask,
        "bad_tokens": bad_tokens,
        "bad_mask": bad_mask,
    }


def collate_hardlabel_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    full_tokens, full_mask = pad_token_batch([item["full_tokens"] for item in items])
    good_tokens, good_mask = pad_token_batch([item["good_tokens"] for item in items])
    bad_tokens, bad_mask = pad_token_batch([item["bad_tokens"] for item in items])
    return {
        "scene_id": [item["scene_id"] for item in items],
        "good_method": [item["good_method"] for item in items],
        "bad_method": [item["bad_method"] for item in items],
        "full_tokens": full_tokens,
        "full_mask": full_mask,
        "good_tokens": good_tokens,
        "good_mask": good_mask,
        "bad_tokens": bad_tokens,
        "bad_mask": bad_mask,
        "target_margin": torch.as_tensor([item["target_margin"] for item in items], dtype=torch.float32),
    }


def pad_token_batch(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_frames = max(int(tensor.shape[0]) for tensor in tensors)
    token_slots = int(tensors[0].shape[1])
    token_dim = int(tensors[0].shape[2])
    batch = torch.zeros((len(tensors), max_frames, token_slots, token_dim), dtype=torch.float32)
    mask = torch.zeros((len(tensors), max_frames), dtype=torch.float32)
    for index, tensor in enumerate(tensors):
        frame_count = int(tensor.shape[0])
        batch[index, :frame_count] = tensor.float()
        mask[index, :frame_count] = 1.0
    return batch, mask


def train_readout(config: TrainConfig) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset = FullTokenSceneDataset(config.train_index)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
        generator=generator,
        drop_last=True,
    )
    first = dataset[0]["tokens"]
    model = PooledReadout(
        token_dim=int(first.shape[-1]),
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        dropout=config.dropout,
    ).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: list[dict[str, Any]] = []
    start_time = time.time()
    total_steps = config.epochs * len(loader)
    global_step = 0
    best_score = -float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in loader:
            global_step += 1
            tokens = batch["tokens"].to(config.device, non_blocking=True)
            batch_size, frame_count = tokens.shape[:2]
            full_mask = torch.ones((batch_size, frame_count), device=config.device)
            ratio = config.subset_ratios[(global_step - 1) % len(config.subset_ratios)]
            good_mask = uniform_mask(batch_size, frame_count, ratio, config.device)
            bad_mask = contiguous_mask(batch_size, frame_count, ratio, config.device)

            z_full = model(tokens, full_mask)
            z_good = model(tokens, good_mask)
            z_bad = model(tokens, bad_mask)

            pos_loss = 1.0 - F.cosine_similarity(z_good, z_full.detach(), dim=-1).mean()
            logits = z_good @ z_full.detach().T / config.nce_temperature
            labels = torch.arange(batch_size, device=config.device)
            nce_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            good_score = F.cosine_similarity(z_good, z_full.detach(), dim=-1)
            bad_score = F.cosine_similarity(z_bad, z_full.detach(), dim=-1)
            rank_loss = F.relu(config.rank_margin - good_score + bad_score).mean()
            loss = (
                config.loss_pos_weight * pos_loss
                + config.loss_nce_weight * nce_loss
                + config.loss_rank_weight * rank_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step == 1 or global_step % 20 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1e-6)
                eta_sec = (total_steps - global_step) / max(steps_per_sec, 1e-6)
                row = {
                    "event": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "total_steps": total_steps,
                    "loss": round(float(loss.detach().cpu()), 6),
                    "pos_loss": round(float(pos_loss.detach().cpu()), 6),
                    "nce_loss": round(float(nce_loss.detach().cpu()), 6),
                    "rank_loss": round(float(rank_loss.detach().cpu()), 6),
                    "steps_per_sec": round(steps_per_sec, 4),
                    "eta_sec": round(eta_sec, 1),
                }
                history.append(row)
                print(json.dumps(row), flush=True)

        eval_summary = {}
        if config.eval_every_epochs and epoch % config.eval_every_epochs == 0:
            eval_summary = evaluate_ltm30(model, config)
            print(json.dumps({"event": "eval", "epoch": epoch, **eval_summary}, sort_keys=True), flush=True)
            mean_primary = float(eval_summary.get("mean_primary_spearman", float("nan")))
            score = -mean_primary if not math.isnan(mean_primary) else -float("inf")
            if score > best_score:
                best_score = score
                save_checkpoint(config.run_dir / "best.pt", model, config, epoch, global_step, eval_summary)

        save_checkpoint(config.run_dir / "last.pt", model, config, epoch, global_step, eval_summary)

    final = {
        "epochs": config.epochs,
        "steps": global_step,
        "elapsed_sec": round(time.time() - start_time, 2),
        "best_expected_alignment": best_score,
    }
    (config.run_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (config.run_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **final}, sort_keys=True), flush=True)
    return final


def train_hardlabel_readout(config: HardLabelTrainConfig) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset = HardLabelPairDataset(
        labels_csv=config.labels_csv,
        pairs_per_scene=config.pairs_per_scene,
        seed=config.seed,
        method_filters=config.method_filters,
        min_target_margin_fraction=config.min_target_margin_fraction,
    )
    (config.run_dir / "pair_summary.json").write_text(json.dumps(dataset.summary, indent=2) + "\n", encoding="utf-8")
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_hardlabel_pairs,
        generator=generator,
        drop_last=True,
        pin_memory=True,
    )
    first = dataset[0]["full_tokens"]
    primary_device = config.devices[0] if config.devices else "cuda:0"
    model: nn.Module = PooledReadout(
        token_dim=int(first.shape[-1]),
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        dropout=config.dropout,
    ).to(primary_device)
    if len(config.devices) > 1 and primary_device.startswith("cuda"):
        device_ids = [int(device.split(":", 1)[1]) for device in config.devices if device.startswith("cuda:")]
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: list[dict[str, Any]] = []
    start_time = time.time()
    total_steps = config.epochs * len(loader)
    global_step = 0
    best_score = -float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in loader:
            global_step += 1
            full_tokens = batch["full_tokens"].to(primary_device, non_blocking=True)
            full_mask = batch["full_mask"].to(primary_device, non_blocking=True)
            good_tokens = batch["good_tokens"].to(primary_device, non_blocking=True)
            good_mask = batch["good_mask"].to(primary_device, non_blocking=True)
            bad_tokens = batch["bad_tokens"].to(primary_device, non_blocking=True)
            bad_mask = batch["bad_mask"].to(primary_device, non_blocking=True)

            z_full = model(full_tokens, full_mask)
            z_good = model(good_tokens, good_mask)
            z_bad = model(bad_tokens, bad_mask)

            good_score = F.cosine_similarity(z_good, z_full.detach(), dim=-1)
            bad_score = F.cosine_similarity(z_bad, z_full.detach(), dim=-1)
            rank_loss = F.relu(config.rank_margin - good_score + bad_score).mean()
            pos_loss = 1.0 - good_score.mean()
            logits = z_good @ z_full.detach().T / config.nce_temperature
            labels = torch.arange(z_good.shape[0], device=primary_device)
            nce_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            loss = (
                config.loss_rank_weight * rank_loss
                + config.loss_pos_weight * pos_loss
                + config.loss_nce_weight * nce_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step == 1 or global_step % 20 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1e-6)
                eta_sec = (total_steps - global_step) / max(steps_per_sec, 1e-6)
                row = {
                    "event": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "total_steps": total_steps,
                    "loss": round(float(loss.detach().cpu()), 6),
                    "rank_loss": round(float(rank_loss.detach().cpu()), 6),
                    "pos_loss": round(float(pos_loss.detach().cpu()), 6),
                    "nce_loss": round(float(nce_loss.detach().cpu()), 6),
                    "good_score": round(float(good_score.mean().detach().cpu()), 6),
                    "bad_score": round(float(bad_score.mean().detach().cpu()), 6),
                    "steps_per_sec": round(steps_per_sec, 4),
                    "eta_sec": round(eta_sec, 1),
                }
                history.append(row)
                print(json.dumps(row), flush=True)

        eval_summary = {}
        if config.eval_every_epochs and epoch % config.eval_every_epochs == 0:
            eval_config = TrainConfig(
                train_index=Path(),
                run_dir=config.run_dir,
                val_metrics_csv=config.val_metrics_csv,
                val_cache_root=config.val_cache_root,
                device=primary_device,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                dropout=config.dropout,
            )
            eval_summary = evaluate_ltm30(model, eval_config)
            print(json.dumps({"event": "eval", "epoch": epoch, **eval_summary}, sort_keys=True), flush=True)
            expected_alignment = float(eval_summary.get("mean_primary_expected_alignment", float("nan")))
            score = expected_alignment if not math.isnan(expected_alignment) else -float("inf")
            if score > best_score:
                best_score = score
                save_hardlabel_checkpoint(config.run_dir / "best.pt", model, config, epoch, global_step, eval_summary)

        save_hardlabel_checkpoint(config.run_dir / "last.pt", model, config, epoch, global_step, eval_summary)

    final = {
        "epochs": config.epochs,
        "steps": global_step,
        "elapsed_sec": round(time.time() - start_time, 2),
        "best_expected_alignment": best_score,
    }
    (config.run_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (config.run_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **final}, sort_keys=True), flush=True)
    return final


def train_attention_multimetric_readout(config: AttentionHardLabelTrainConfig) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset = MultiMetricHardLabelPairDataset(
        labels_csv=config.labels_csv,
        metrics=config.metrics,
        pairs_per_scene_metric=config.pairs_per_scene_metric,
        seed=config.seed,
        method_filters=config.method_filters,
        min_metric_margin_fraction=config.min_metric_margin_fraction,
    )
    (config.run_dir / "pair_summary.json").write_text(json.dumps(dataset.summary, indent=2) + "\n", encoding="utf-8")
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_multimetric_pairs,
        generator=generator,
        drop_last=True,
        pin_memory=True,
    )
    first = dataset[0]["full_tokens"]
    primary_device = config.devices[0] if config.devices else "cuda:0"
    model: nn.Module = CrossAttentionReadout(
        token_dim=int(first.shape[-1]),
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_metrics=len(config.metrics),
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(primary_device)
    if len(config.devices) > 1 and primary_device.startswith("cuda"):
        device_ids = [int(device.split(":", 1)[1]) for device in config.devices if device.startswith("cuda:")]
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: list[dict[str, Any]] = []
    start_time = time.time()
    total_steps = config.epochs * len(loader)
    global_step = 0
    best_score = -float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in loader:
            global_step += 1
            full_tokens = batch["full_tokens"].to(primary_device, non_blocking=True)
            full_mask = batch["full_mask"].to(primary_device, non_blocking=True)
            good_tokens = batch["good_tokens"].to(primary_device, non_blocking=True)
            good_mask = batch["good_mask"].to(primary_device, non_blocking=True)
            bad_tokens = batch["bad_tokens"].to(primary_device, non_blocking=True)
            bad_mask = batch["bad_mask"].to(primary_device, non_blocking=True)
            metric_index = batch["metric_index"].to(primary_device, non_blocking=True)

            z_full, full_scores = model(full_tokens, full_mask, return_scores=True)
            z_good, good_scores = model(good_tokens, good_mask, return_scores=True)
            z_bad, bad_scores = model(bad_tokens, bad_mask, return_scores=True)

            gather_index = metric_index[:, None]
            full_metric_score = full_scores.gather(1, gather_index).squeeze(1)
            good_metric_score = good_scores.gather(1, gather_index).squeeze(1)
            bad_metric_score = bad_scores.gather(1, gather_index).squeeze(1)
            metric_rank_loss = F.relu(config.rank_margin - good_metric_score + bad_metric_score).mean()
            full_rank_loss = F.relu(config.rank_margin - full_metric_score + good_metric_score).mean()

            good_embed_score = F.cosine_similarity(z_good, z_full.detach(), dim=-1)
            bad_embed_score = F.cosine_similarity(z_bad, z_full.detach(), dim=-1)
            embedding_rank_loss = F.relu(0.05 - good_embed_score + bad_embed_score).mean()
            embedding_pos_loss = 1.0 - good_embed_score.mean()
            logits = z_good @ z_full.detach().T / config.nce_temperature
            labels = torch.arange(z_good.shape[0], device=primary_device)
            nce_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

            loss = (
                metric_rank_loss
                + config.full_rank_weight * full_rank_loss
                + config.embedding_rank_weight * embedding_rank_loss
                + config.embedding_pos_weight * embedding_pos_loss
                + config.nce_weight * nce_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step == 1 or global_step % 20 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1e-6)
                eta_sec = (total_steps - global_step) / max(steps_per_sec, 1e-6)
                row = {
                    "event": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "total_steps": total_steps,
                    "loss": round(float(loss.detach().cpu()), 6),
                    "metric_rank_loss": round(float(metric_rank_loss.detach().cpu()), 6),
                    "full_rank_loss": round(float(full_rank_loss.detach().cpu()), 6),
                    "embedding_rank_loss": round(float(embedding_rank_loss.detach().cpu()), 6),
                    "embedding_pos_loss": round(float(embedding_pos_loss.detach().cpu()), 6),
                    "nce_loss": round(float(nce_loss.detach().cpu()), 6),
                    "good_metric_score": round(float(good_metric_score.mean().detach().cpu()), 6),
                    "bad_metric_score": round(float(bad_metric_score.mean().detach().cpu()), 6),
                    "steps_per_sec": round(steps_per_sec, 4),
                    "eta_sec": round(eta_sec, 1),
                }
                history.append(row)
                print(json.dumps(row), flush=True)

        eval_summary = {}
        if config.eval_every_epochs and epoch % config.eval_every_epochs == 0:
            eval_summary = evaluate_ltm30_multimetric(
                model=model,
                run_dir=config.run_dir,
                val_metrics_csv=config.val_metrics_csv,
                val_cache_root=config.val_cache_root,
                device=primary_device,
                metrics=config.metrics,
            )
            print(json.dumps({"event": "eval", "epoch": epoch, **eval_summary}, sort_keys=True), flush=True)
            expected_alignment = float(eval_summary.get("mean_primary_expected_alignment", float("nan")))
            score = expected_alignment if not math.isnan(expected_alignment) else -float("inf")
            if score > best_score:
                best_score = score
                save_attention_checkpoint(config.run_dir / "best.pt", model, config, epoch, global_step, eval_summary)

        save_attention_checkpoint(config.run_dir / "last.pt", model, config, epoch, global_step, eval_summary)

    final = {
        "epochs": config.epochs,
        "steps": global_step,
        "elapsed_sec": round(time.time() - start_time, 2),
        "best_expected_alignment": best_score,
    }
    (config.run_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (config.run_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **final}, sort_keys=True), flush=True)
    return final


def uniform_mask(batch_size: int, frame_count: int, ratio: float, device: str) -> torch.Tensor:
    k = max(1, int(round(frame_count * ratio)))
    base = torch.linspace(0, frame_count - 1, steps=k, device=device).round().long().unique()
    if base.numel() < k:
        fill = torch.arange(frame_count, device=device)
        missing = fill[~torch.isin(fill, base)][: k - base.numel()]
        base = torch.sort(torch.cat([base, missing]))[0]
    mask = torch.zeros((batch_size, frame_count), device=device)
    mask[:, base[:k]] = 1.0
    return mask


def contiguous_mask(batch_size: int, frame_count: int, ratio: float, device: str) -> torch.Tensor:
    k = max(1, int(round(frame_count * ratio)))
    starts = torch.randint(0, max(frame_count - k + 1, 1), (batch_size,), device=device)
    offsets = torch.arange(k, device=device)
    indices = starts[:, None] + offsets[None, :]
    mask = torch.zeros((batch_size, frame_count), device=device)
    mask.scatter_(1, indices.clamp(max=frame_count - 1), 1.0)
    return mask


def evaluate_ltm30(model: nn.Module, config: TrainConfig) -> dict[str, Any]:
    if not config.val_metrics_csv.is_file():
        return {"val_available": False}
    rows = read_csv(config.val_metrics_csv)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scene_id"], []).append(row)

    model.eval()
    scored_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for scene_id, scene_rows in sorted(grouped.items()):
            full_path = config.val_cache_root / scene_id / "full" / "camera_and_register_tokens.pt"
            if not full_path.is_file():
                continue
            full_tokens = load_cache_tokens(full_path).to(config.device)
            full_mask = torch.ones((1, full_tokens.shape[1]), device=config.device)
            z_full = model(full_tokens, full_mask)
            for row in scene_rows:
                method = row["method"]
                sub_path = config.val_cache_root / scene_id / method / "camera_and_register_tokens.pt"
                if not sub_path.is_file():
                    continue
                sub_tokens = load_cache_tokens(sub_path).to(config.device)
                sub_mask = torch.ones((1, sub_tokens.shape[1]), device=config.device)
                z_sub = model(sub_tokens, sub_mask)
                score = F.cosine_similarity(z_sub, z_full, dim=-1).item()
                scored = dict(row)
                scored["readout_cosine"] = score
                scored_rows.append(scored)

    metric_summaries: dict[str, float] = {}
    primary_values = []
    for metric in PRIMARY_VAL_METRICS:
        correlations = []
        for scene_id in sorted({row["scene_id"] for row in scored_rows}):
            scene_rows = [row for row in scored_rows if row["scene_id"] == scene_id]
            xs = [float(row["readout_cosine"]) for row in scene_rows]
            ys = [float(row[metric]) for row in scene_rows if row.get(metric, "") != ""]
            if len(xs) == len(ys) and len(xs) >= 3:
                rho = spearman(xs, ys)
                if not math.isnan(rho):
                    correlations.append(rho)
        value = sum(correlations) / len(correlations) if correlations else float("nan")
        metric_summaries[f"{metric}_mean_spearman"] = value
        if not math.isnan(value):
            primary_values.append(value)

    out_csv = config.run_dir / "ltm30_readout_scores.csv"
    write_csv(out_csv, scored_rows)
    mean_primary = sum(primary_values) / len(primary_values) if primary_values else float("nan")
    return {
        "val_available": True,
        "val_rows": len(scored_rows),
        "mean_primary_spearman": mean_primary,
        "mean_primary_expected_alignment": -mean_primary if not math.isnan(mean_primary) else float("nan"),
        **metric_summaries,
    }


def evaluate_ltm30_multimetric(
    *,
    model: nn.Module,
    run_dir: Path,
    val_metrics_csv: Path,
    val_cache_root: Path,
    device: str,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    if not val_metrics_csv.is_file():
        return {"val_available": False}
    rows = read_csv(val_metrics_csv)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scene_id"], []).append(row)

    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    eval_model.eval()
    scored_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for scene_id, scene_rows in sorted(grouped.items()):
            full_path = val_cache_root / scene_id / "full" / "camera_and_register_tokens.pt"
            if not full_path.is_file():
                continue
            full_tokens = load_cache_tokens(full_path).to(device)
            full_mask = torch.ones((1, full_tokens.shape[1]), device=device)
            z_full, full_scores = eval_model(full_tokens, full_mask, return_scores=True)
            for row in scene_rows:
                method = row["method"]
                sub_path = val_cache_root / scene_id / method / "camera_and_register_tokens.pt"
                if not sub_path.is_file():
                    continue
                sub_tokens = load_cache_tokens(sub_path).to(device)
                sub_mask = torch.ones((1, sub_tokens.shape[1]), device=device)
                z_sub, sub_scores = eval_model(sub_tokens, sub_mask, return_scores=True)
                scored = dict(row)
                scored["readout_cosine"] = F.cosine_similarity(z_sub, z_full, dim=-1).item()
                for metric_index, metric in enumerate(metrics):
                    scored[f"{metric}_score"] = float(sub_scores[0, metric_index].detach().cpu())
                    scored[f"{metric}_full_score"] = float(full_scores[0, metric_index].detach().cpu())
                scored_rows.append(scored)

    metric_summaries: dict[str, float] = {}
    head_primary_values = []
    embedding_primary_values = []
    for metric in metrics:
        head_correlations = []
        embedding_correlations = []
        for scene_id in sorted({row["scene_id"] for row in scored_rows}):
            scene_rows = [row for row in scored_rows if row["scene_id"] == scene_id]
            ys = [float(row[metric]) for row in scene_rows if row.get(metric, "") != ""]
            head_xs = [float(row[f"{metric}_score"]) for row in scene_rows]
            embedding_xs = [float(row["readout_cosine"]) for row in scene_rows]
            if len(head_xs) == len(ys) and len(ys) >= 3:
                rho = spearman(head_xs, ys)
                if not math.isnan(rho):
                    head_correlations.append(rho)
            if len(embedding_xs) == len(ys) and len(ys) >= 3:
                rho = spearman(embedding_xs, ys)
                if not math.isnan(rho):
                    embedding_correlations.append(rho)
        head_value = sum(head_correlations) / len(head_correlations) if head_correlations else float("nan")
        embedding_value = (
            sum(embedding_correlations) / len(embedding_correlations) if embedding_correlations else float("nan")
        )
        metric_summaries[f"{metric}_head_mean_spearman"] = head_value
        metric_summaries[f"{metric}_embedding_mean_spearman"] = embedding_value
        if not math.isnan(head_value):
            head_primary_values.append(head_value)
        if not math.isnan(embedding_value):
            embedding_primary_values.append(embedding_value)

    out_csv = run_dir / "ltm30_multimetric_scores.csv"
    write_csv(out_csv, scored_rows)
    head_mean = sum(head_primary_values) / len(head_primary_values) if head_primary_values else float("nan")
    embedding_mean = (
        sum(embedding_primary_values) / len(embedding_primary_values) if embedding_primary_values else float("nan")
    )
    return {
        "val_available": True,
        "val_rows": len(scored_rows),
        "mean_primary_spearman": head_mean,
        "mean_primary_expected_alignment": -head_mean if not math.isnan(head_mean) else float("nan"),
        "embedding_mean_primary_spearman": embedding_mean,
        "embedding_mean_primary_expected_alignment": -embedding_mean if not math.isnan(embedding_mean) else float("nan"),
        **metric_summaries,
    }


def load_cache_tokens(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu").float()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor[None]


def load_token_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu").float()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected token tensor [N,T,C] from {path}, got {tuple(tensor.shape)}")
    return tensor


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_rows_by_method(rows: list[dict[str, str]], method_filters: tuple[str, ...]) -> list[dict[str, str]]:
    filters = tuple(item for item in method_filters if item)
    if not filters:
        return rows
    return [row for row in rows if any(item in row["method"] for item in filters)]


def metric_margin_threshold(values: list[float], min_margin_fraction: float) -> float:
    if not values:
        return float("inf")
    spread = max(values) - min(values)
    if min_margin_fraction <= 0.0 or spread <= 0.0:
        return 1e-8
    return max(1e-8, min_margin_fraction * spread)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: list[float]) -> list[float]:
    pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0
        for _value, index in pairs[i:j]:
            ranks[index] = avg
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(vx * vy)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    epoch: int,
    global_step: int,
    eval_summary: dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "config": {
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "dropout": config.dropout,
        },
        "epoch": epoch,
        "global_step": global_step,
        "eval_summary": eval_summary,
    }
    torch.save(payload, path)


def save_hardlabel_checkpoint(
    path: Path,
    model: nn.Module,
    config: HardLabelTrainConfig,
    epoch: int,
    global_step: int,
    eval_summary: dict[str, Any],
) -> None:
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    payload = {
        "model": raw_model.state_dict(),
        "config": {
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "dropout": config.dropout,
            "devices": list(config.devices),
            "method_filters": list(config.method_filters),
            "min_target_margin_fraction": config.min_target_margin_fraction,
            "target": "hard_native_label_pairwise_ranking",
        },
        "epoch": epoch,
        "global_step": global_step,
        "eval_summary": eval_summary,
    }
    torch.save(payload, path)


def save_attention_checkpoint(
    path: Path,
    model: nn.Module,
    config: AttentionHardLabelTrainConfig,
    epoch: int,
    global_step: int,
    eval_summary: dict[str, Any],
) -> None:
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    payload = {
        "model": raw_model.state_dict(),
        "config": {
            "model_type": "cross_attention_multimetric",
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "metrics": list(config.metrics),
            "devices": list(config.devices),
            "method_filters": list(config.method_filters),
            "min_metric_margin_fraction": config.min_metric_margin_fraction,
            "target": "hard_native_label_metric_head_pairwise_ranking",
        },
        "epoch": epoch,
        "global_step": global_step,
        "eval_summary": eval_summary,
    }
    torch.save(payload, path)
