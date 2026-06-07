"""Training utilities for Stage 2.0 readout calibration."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from vggt_omega_selector.readouts.models import PooledReadout


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


def load_cache_tokens(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu").float()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor[None]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
