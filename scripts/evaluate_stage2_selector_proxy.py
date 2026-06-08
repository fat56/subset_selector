#!/usr/bin/env python3
"""Evaluate Stage 2 selector checkpoints against cache-only proxy baselines."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stage2_selector_training import SelectorFeatureDataset, collate_selector_batch, fixed_k  # noqa: E402
from vggt_omega_selector.selectors.models import FixedKSetSelector, hard_topk_mask  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a selector checkpoint on cached mean-register proxy metrics.")
    parser.add_argument("--feature-index", default="runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/feature_index.json")
    parser.add_argument("--checkpoint", default="runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/best_hard_proxy.pt")
    parser.add_argument("--out", default="runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/proxy_eval.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--ratio", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-seeds", default="0,1,2,3,4")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    records = json.loads(resolve(args.feature_index).read_text(encoding="utf-8"))["records"]
    dataset = SelectorFeatureDataset(records, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_selector_batch,
        pin_memory=device.type == "cuda",
    )

    sample = dataset[0]
    checkpoint_path = resolve(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FixedKSetSelector(
        input_dim=int(sample["features"].shape[-1]),
        hidden_dim=int(checkpoint["config"].get("hidden_dim", 512)),
        num_layers=int(checkpoint["config"].get("num_layers", 4)),
        num_heads=int(checkpoint["config"].get("num_heads", 8)),
        dropout=float(checkpoint["config"].get("dropout", 0.1)),
        max_frames=max(int(record["frame_count"]) for record in records),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    random_seeds = [int(seed) for seed in args.random_seeds.split(",") if seed.strip()]
    metrics = evaluate_methods(model, loader, device, args.ratio, random_seeds)
    payload = {
        "feature_index": str(resolve(args.feature_index)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_step": int(checkpoint["step"]),
        "split": args.split,
        "ratio": args.ratio,
        "rows": len(dataset),
        "metrics": metrics,
    }
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def evaluate_methods(
    model: FixedKSetSelector,
    loader: DataLoader[Any],
    device: torch.device,
    ratio: float,
    random_seeds: list[int],
) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {
        "learned_topk": [],
        "uniform_stride": [],
        "prefix": [],
        "all_frames": [],
    }
    random_values: dict[int, list[float]] = {seed: [] for seed in random_seeds}

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            features = batch["features"].to(device, non_blocking=True)
            register_mean = batch["register_mean"].to(device, non_blocking=True)
            frame_mask = batch["frame_mask"].to(device, non_blocking=True)
            full_embedding = F.normalize(batch["full_embedding"].to(device, non_blocking=True), dim=-1)
            k = fixed_k(frame_mask, ratio)

            scores = model(features, frame_mask)
            values["learned_topk"].extend(cosine_for_mask(register_mean, hard_topk_mask(scores, frame_mask, k), full_embedding))
            values["uniform_stride"].extend(cosine_for_mask(register_mean, uniform_mask(frame_mask, k), full_embedding))
            values["prefix"].extend(cosine_for_mask(register_mean, prefix_mask(frame_mask, k), full_embedding))
            values["all_frames"].extend(cosine_for_mask(register_mean, frame_mask.float(), full_embedding))
            for seed in random_seeds:
                random_values[seed].extend(cosine_for_mask(register_mean, random_mask(frame_mask, k, seed, batch_index), full_embedding))

    metrics = {name: summarize(metric_values) for name, metric_values in values.items()}
    random_summaries = {str(seed): summarize(metric_values) for seed, metric_values in random_values.items()}
    random_all = [value for metric_values in random_values.values() for value in metric_values]
    metrics["random"] = {
        **summarize(random_all),
        "seed_mean_min": min(summary["mean"] for summary in random_summaries.values()),
        "seed_mean_max": max(summary["mean"] for summary in random_summaries.values()),
        "seeds": len(random_summaries),
    }
    return metrics


def cosine_for_mask(register_mean: torch.Tensor, mask: torch.Tensor, full_embedding: torch.Tensor) -> list[float]:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pooled = F.normalize((register_mean * mask[..., None]).sum(dim=1) / denom, dim=-1)
    return F.cosine_similarity(pooled, full_embedding, dim=-1).detach().cpu().tolist()


def uniform_mask(frame_mask: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(frame_mask, dtype=torch.float32)
    counts = frame_mask.sum(dim=1).long()
    for row in range(frame_mask.shape[0]):
        n = int(counts[row].item())
        count = max(1, min(int(k[row].item()), n))
        if count == 1:
            indices = [0]
        else:
            indices = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
            fill = 0
            while len(indices) < count:
                if fill not in indices:
                    indices.append(fill)
                fill += 1
            indices = sorted(indices[:count])
        out[row, torch.tensor(indices, device=frame_mask.device)] = 1.0
    return out


def prefix_mask(frame_mask: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(frame_mask, dtype=torch.float32)
    counts = frame_mask.sum(dim=1).long()
    for row in range(frame_mask.shape[0]):
        count = max(1, min(int(k[row].item()), int(counts[row].item())))
        out[row, :count] = 1.0
    return out


def random_mask(frame_mask: torch.Tensor, k: torch.Tensor, seed: int, batch_index: int) -> torch.Tensor:
    out = torch.zeros_like(frame_mask, dtype=torch.float32)
    counts = frame_mask.sum(dim=1).long()
    for row in range(frame_mask.shape[0]):
        n = int(counts[row].item())
        count = max(1, min(int(k[row].item()), n))
        rng = random.Random((seed + 1) * 1_000_003 + batch_index * 10_007 + row)
        indices = rng.sample(range(n), count)
        out[row, torch.tensor(indices, device=frame_mask.device)] = 1.0
    return out


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "count": int(tensor.numel()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
