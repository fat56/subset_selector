#!/usr/bin/env python3
"""Cache compact selector features and train the first Stage 2 fixed-ratio selector."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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

from vggt_omega_selector.backbones.vggt_omega import VGGTOmegaIntegration  # noqa: E402
from vggt_omega_selector.selectors.models import FixedKSetSelector, hard_topk_mask, soft_topk_mask  # noqa: E402


@dataclass
class SelectorTrainConfig:
    feature_index: Path
    run_dir: Path
    train_devices: list[str]
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    ratio: float = 0.20
    temperature_start: float = 1.0
    temperature_end: float = 0.2
    objective: str = "meanpool"
    pos_weight: float = 1.0
    nce_weight: float = 0.2
    rank_weight: float = 0.0
    uniform_ce_weight: float = 0.0
    rank_margin: float = 0.0
    num_workers: int = 2
    eval_every_epochs: int = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 0004 Stage 2 fixed-ratio selector training.")
    parser.add_argument("--manifest", default="docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_manifest.json")
    parser.add_argument("--cache-root", default="caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512")
    parser.add_argument("--run-dir", default="runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector")
    parser.add_argument("--checkpoint", default="512")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--cache-devices", default="cuda:0,cuda:1")
    parser.add_argument("--train-devices", default="cuda:0,cuda:1")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ratio", type=float, default=0.20)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.2)
    parser.add_argument("--objective", choices=["meanpool", "baseline_rank"], default="meanpool")
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--nce-weight", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--uniform-ce-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)

    manifest_path = resolve(args.manifest)
    cache_root = resolve(args.cache_root)
    run_dir = resolve(args.run_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs, records = build_feature_jobs(manifest, cache_root, run_dir, args.max_scenes)
    (run_dir / "cache_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    if not args.skip_cache:
        missing = [job for job in jobs if args.force_cache or not feature_ready(Path(job["output_dir"]))]
        if missing:
            run_feature_cache_parallel(
                jobs=missing,
                run_dir=run_dir,
                checkpoint=args.checkpoint,
                image_resolution=args.image_resolution,
                mode=args.mode,
                feature_dtype=args.feature_dtype,
                devices=[device.strip() for device in args.cache_devices.split(",") if device.strip()],
                force=args.force_cache,
            )
        else:
            print(json.dumps({"event": "cache_skip", "jobs": len(jobs)}), flush=True)

    feature_index = run_dir / "feature_index.json"
    validate_and_write_index(records, feature_index)
    if args.cache_only:
        print(json.dumps({"event": "cache_only_done", "records": len(records)}, indent=2), flush=True)
        return 0

    config = SelectorTrainConfig(
        feature_index=feature_index,
        run_dir=run_dir,
        train_devices=[device.strip() for device in args.train_devices.split(",") if device.strip()],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        ratio=args.ratio,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        objective=args.objective,
        pos_weight=args.pos_weight,
        nce_weight=args.nce_weight,
        rank_weight=args.rank_weight,
        uniform_ce_weight=args.uniform_ce_weight,
        rank_margin=args.rank_margin,
        num_workers=args.num_workers,
    )
    train_selector(config)
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_feature_jobs(
    manifest: dict[str, Any],
    cache_root: Path,
    run_dir: Path,
    max_scenes: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenes = manifest["scenes"][:max_scenes] if max_scenes else manifest["scenes"]
    jobs = []
    records = []
    image_list_dir = run_dir / "image_lists"
    image_list_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        scene_id = scene["scene_id"]
        image_paths = [str(resolve(frame["image_path"])) for frame in scene["frames"]]
        frame_ids = [str(frame["frame_id"]) for frame in scene["frames"]]
        image_list = image_list_dir / f"{scene_id}.txt"
        image_list.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        output_dir = cache_root / scene_id
        job = {
            "id": scene_id,
            "scene_id": scene_id,
            "scene_key": scene["scene_key"],
            "dataset": scene["dataset"],
            "split": scene["split"],
            "image_count": len(image_paths),
            "frame_ids": frame_ids,
            "image_list": str(image_list.resolve()),
            "output_dir": str(output_dir.resolve()),
        }
        jobs.append(job)
        records.append(
            {
                "scene_id": scene_id,
                "scene_key": scene["scene_key"],
                "dataset": scene["dataset"],
                "split": scene["split"],
                "frame_count": len(image_paths),
                "feature_path": str((output_dir / "selector_features.pt").resolve()),
            }
        )
    return jobs, records


def feature_ready(output_dir: Path) -> bool:
    return (output_dir / "selector_features.pt").is_file() and (output_dir / "manifest.json").is_file()


def run_feature_cache_parallel(
    *,
    jobs: list[dict[str, Any]],
    run_dir: Path,
    checkpoint: str,
    image_resolution: int,
    mode: str,
    feature_dtype: str,
    devices: list[str],
    force: bool,
) -> None:
    if not devices:
        raise ValueError("At least one cache device is required.")
    integration = VGGTOmegaIntegration.discover()
    shards = [[] for _ in devices]
    for index, job in enumerate(jobs):
        shards[index % len(shards)].append(job)

    processes: list[tuple[str, subprocess.Popen[Any], Any]] = []
    for device, shard in zip(devices, shards, strict=True):
        if not shard:
            continue
        device_id = device.replace(":", "")
        jobs_path = run_dir / f"feature_jobs_{device_id}.json"
        jobs_path.write_text(json.dumps(shard, indent=2) + "\n", encoding="utf-8")
        log_path = run_dir / f"feature_cache_{device_id}.log"
        command = [
            str(integration.python),
            "-m",
            "vggt_omega_selector.tools.vggt_selector_feature_runner",
            "--vggt-root",
            str(integration.root),
            "--checkpoint",
            str(integration.checkpoint_path(checkpoint)),
            "--jobs-json",
            str(jobs_path),
            "--image-resolution",
            str(image_resolution),
            "--mode",
            mode,
            "--device",
            device,
            "--feature-dtype",
            feature_dtype,
        ]
        if force:
            command.append("--force")
        print(json.dumps({"event": "feature_cache_start", "device": device, "jobs": len(shard), "log": str(log_path)}), flush=True)
        log_file = log_path.open("w", encoding="utf-8")
        processes.append(
            (
                device,
                subprocess.Popen(
                    command,
                    cwd=integration.project_root,
                    env=integration.subprocess_env(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
                log_file,
            )
        )

    failed = []
    for device, process, log_file in processes:
        returncode = process.wait()
        log_file.close()
        print(json.dumps({"event": "feature_cache_done", "device": device, "returncode": returncode}), flush=True)
        if returncode != 0:
            failed.append((device, returncode))
    if failed:
        raise SystemExit(f"Feature cache failed: {failed}")


def validate_and_write_index(records: list[dict[str, Any]], path: Path) -> None:
    missing = [record["feature_path"] for record in records if not Path(record["feature_path"]).is_file()]
    if missing:
        raise RuntimeError(f"Missing selector feature caches ({len(missing)}):\n" + "\n".join(missing[:20]))
    path.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")


class SelectorFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], split: str) -> None:
        self.records = [record for record in records if record["split"] == split]
        if not self.records:
            raise ValueError(f"No records for split={split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        payload = torch.load(record["feature_path"], map_location="cpu")
        return {
            "scene_id": record["scene_id"],
            "dataset": record["dataset"],
            "features": payload["frame_features"].float(),
            "register_mean": payload["register_mean"].float(),
            "full_embedding": payload["full_embedding"].float(),
        }


def collate_selector_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_frames = max(item["features"].shape[0] for item in items)
    feature_dim = items[0]["features"].shape[-1]
    reg_dim = items[0]["register_mean"].shape[-1]
    batch = len(items)
    features = torch.zeros((batch, max_frames, feature_dim), dtype=torch.float32)
    register_mean = torch.zeros((batch, max_frames, reg_dim), dtype=torch.float32)
    frame_mask = torch.zeros((batch, max_frames), dtype=torch.bool)
    full_embedding = torch.stack([item["full_embedding"] for item in items])
    for row, item in enumerate(items):
        n = item["features"].shape[0]
        features[row, :n] = item["features"]
        register_mean[row, :n] = item["register_mean"]
        frame_mask[row, :n] = True
    return {
        "features": features,
        "register_mean": register_mean,
        "frame_mask": frame_mask,
        "full_embedding": full_embedding,
        "scene_ids": [item["scene_id"] for item in items],
        "datasets": [item["dataset"] for item in items],
    }


def train_selector(config: SelectorTrainConfig) -> dict[str, Any]:
    payload = json.loads(config.feature_index.read_text(encoding="utf-8"))
    records = payload["records"]
    train_ds = SelectorFeatureDataset(records, "train")
    val_ds = SelectorFeatureDataset(records, "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_selector_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_selector_batch,
        pin_memory=True,
    )

    sample = train_ds[0]
    input_dim = int(sample["features"].shape[-1])
    max_frames = max(int(record["frame_count"]) for record in records)
    model = FixedKSetSelector(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
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
    best_soft = -1.0
    best_hard = -1.0
    best_margin = -float("inf")
    started = time.time()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "train_config.json").write_text(json.dumps(asdict(config), default=str, indent=2) + "\n", encoding="utf-8")

    for epoch in range(1, config.epochs + 1):
        model.train()
        temperature = interpolate(config.temperature_start, config.temperature_end, epoch, config.epochs)
        for batch in train_loader:
            step += 1
            loss, stats = selector_loss(model, batch, device, config, temperature, step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 20 == 0:
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
                            "temperature": round(float(temperature), 4),
                            "soft_cosine": round(stats["soft_cosine"], 6),
                            "hard_proxy_cosine": round(stats["hard_proxy_cosine"], 6),
                            "uniform_proxy_cosine": round(stats["uniform_proxy_cosine"], 6),
                            "hard_minus_uniform": round(stats["hard_minus_uniform"], 6),
                            "nce_loss": round(stats["nce_loss"], 6),
                            "rank_loss": round(stats["rank_loss"], 6),
                            "uniform_ce_loss": round(stats["uniform_ce_loss"], 6),
                            "steps_per_sec": round(steps_per_sec, 4),
                            "eta_sec": round(eta, 1),
                        }
                    ),
                    flush=True,
                )

        if epoch % config.eval_every_epochs == 0:
            metrics = evaluate_selector(model, val_loader, device, config.ratio)
            print(json.dumps({"event": "eval", "epoch": epoch, **metrics}), flush=True)
            if metrics["soft_cosine"] > best_soft:
                best_soft = metrics["soft_cosine"]
                save_checkpoint(config.run_dir / "best_soft.pt", model, config, epoch, step, metrics)
            if metrics["hard_proxy_cosine"] > best_hard:
                best_hard = metrics["hard_proxy_cosine"]
                save_checkpoint(config.run_dir / "best_hard_proxy.pt", model, config, epoch, step, metrics)
            if metrics.get("hard_minus_uniform", -float("inf")) > best_margin:
                best_margin = metrics["hard_minus_uniform"]
                save_checkpoint(config.run_dir / "best_margin.pt", model, config, epoch, step, metrics)
            save_checkpoint(config.run_dir / "last.pt", model, config, epoch, step, metrics)

    summary = {
        "epochs": config.epochs,
        "steps": step,
        "elapsed_sec": round(time.time() - started, 2),
        "best_soft_cosine": best_soft,
        "best_hard_proxy_cosine": best_hard,
        "best_hard_minus_uniform": best_margin,
        "objective": config.objective,
    }
    (config.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **summary}), flush=True)
    return summary


def selector_loss(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    config: SelectorTrainConfig,
    temperature: float,
    step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = batch["features"].to(device, non_blocking=True)
    register_mean = batch["register_mean"].to(device, non_blocking=True)
    frame_mask = batch["frame_mask"].to(device, non_blocking=True)
    full_embedding = F.normalize(batch["full_embedding"].to(device, non_blocking=True), dim=-1)
    scores = model(features, frame_mask)
    k = fixed_k(frame_mask, config.ratio)
    soft_mask = soft_topk_mask(scores, frame_mask, k, temperature)
    z_soft = aggregate_register(register_mean, soft_mask)
    soft_cosine = F.cosine_similarity(z_soft, full_embedding, dim=-1)
    pos_loss = (1.0 - soft_cosine).mean()
    nce = symmetric_nce(z_soft, full_embedding) if config.nce_weight > 0 else torch.zeros((), device=device)

    uniform = uniform_mask(frame_mask, k)
    random = random_mask(frame_mask, k, seed=step)
    z_uniform = aggregate_register(register_mean, uniform)
    z_random = aggregate_register(register_mean, random)
    uniform_cosine = F.cosine_similarity(z_uniform, full_embedding, dim=-1)
    random_cosine = F.cosine_similarity(z_random, full_embedding, dim=-1)

    if config.objective == "baseline_rank":
        baseline_target = torch.maximum(uniform_cosine, random_cosine).detach()
        rank_loss = F.relu(baseline_target + config.rank_margin - soft_cosine).mean()
        uniform_ce = topk_target_cross_entropy(scores, frame_mask, uniform, k)
    else:
        rank_loss = torch.zeros((), device=device)
        uniform_ce = torch.zeros((), device=device)

    loss = (
        config.pos_weight * pos_loss
        + config.nce_weight * nce
        + config.rank_weight * rank_loss
        + config.uniform_ce_weight * uniform_ce
    )

    with torch.no_grad():
        hard_mask = hard_topk_mask(scores, frame_mask, k)
        z_hard = aggregate_register(register_mean, hard_mask)
        hard_cosine = F.cosine_similarity(z_hard, full_embedding, dim=-1)
        stats = {
            "soft_cosine": float(soft_cosine.mean().item()),
            "hard_proxy_cosine": float(hard_cosine.mean().item()),
            "uniform_proxy_cosine": float(uniform_cosine.mean().item()),
            "hard_minus_uniform": float((hard_cosine - uniform_cosine).mean().item()),
            "nce_loss": float(nce.item()),
            "rank_loss": float(rank_loss.item()),
            "uniform_ce_loss": float(uniform_ce.item()),
        }
    return loss, stats


def evaluate_selector(model: nn.Module, loader: DataLoader[Any], device: torch.device, ratio: float) -> dict[str, float]:
    model.eval()
    soft_values = []
    hard_values = []
    uniform_values = []
    random_values = []
    selected_fracs = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            features = batch["features"].to(device, non_blocking=True)
            register_mean = batch["register_mean"].to(device, non_blocking=True)
            frame_mask = batch["frame_mask"].to(device, non_blocking=True)
            full_embedding = F.normalize(batch["full_embedding"].to(device, non_blocking=True), dim=-1)
            scores = model(features, frame_mask)
            k = fixed_k(frame_mask, ratio)
            soft_mask = soft_topk_mask(scores, frame_mask, k, temperature=0.2)
            hard_mask = hard_topk_mask(scores, frame_mask, k)
            uniform = uniform_mask(frame_mask, k)
            random = random_mask(frame_mask, k, seed=batch_index)
            z_soft = aggregate_register(register_mean, soft_mask)
            z_hard = aggregate_register(register_mean, hard_mask)
            z_uniform = aggregate_register(register_mean, uniform)
            z_random = aggregate_register(register_mean, random)
            soft_values.extend(F.cosine_similarity(z_soft, full_embedding, dim=-1).detach().cpu().tolist())
            hard_values.extend(F.cosine_similarity(z_hard, full_embedding, dim=-1).detach().cpu().tolist())
            uniform_values.extend(F.cosine_similarity(z_uniform, full_embedding, dim=-1).detach().cpu().tolist())
            random_values.extend(F.cosine_similarity(z_random, full_embedding, dim=-1).detach().cpu().tolist())
            selected_fracs.extend((k.float() / frame_mask.sum(dim=1).float()).detach().cpu().tolist())
    hard_mean = float(sum(hard_values) / len(hard_values))
    uniform_mean = float(sum(uniform_values) / len(uniform_values))
    random_mean = float(sum(random_values) / len(random_values))
    return {
        "soft_cosine": float(sum(soft_values) / len(soft_values)),
        "hard_proxy_cosine": hard_mean,
        "uniform_proxy_cosine": uniform_mean,
        "random_proxy_cosine": random_mean,
        "hard_minus_uniform": hard_mean - uniform_mean,
        "hard_minus_random": hard_mean - random_mean,
        "soft_hard_gap": float((sum(soft_values) - sum(hard_values)) / len(soft_values)),
        "mean_selected_fraction": float(sum(selected_fracs) / len(selected_fracs)),
        "val_rows": len(soft_values),
    }


def fixed_k(frame_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    counts = frame_mask.sum(dim=1)
    return torch.clamp(torch.round(counts.float() * ratio).long(), min=1)


def aggregate_register(register_mean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pooled = (register_mean * mask[..., None]).sum(dim=1) / denom
    return F.normalize(pooled, dim=-1)


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


def random_mask(frame_mask: torch.Tensor, k: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.zeros_like(frame_mask, dtype=torch.float32)
    counts = frame_mask.sum(dim=1).long()
    for row in range(frame_mask.shape[0]):
        n = int(counts[row].item())
        count = max(1, min(int(k[row].item()), n))
        generator = torch.Generator(device="cpu")
        generator.manual_seed((seed + 1) * 1_000_003 + row)
        indices = torch.randperm(n, generator=generator)[:count].to(device=frame_mask.device)
        out[row, indices] = 1.0
    return out


def topk_target_cross_entropy(scores: torch.Tensor, frame_mask: torch.Tensor, target_mask: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    valid = frame_mask.to(dtype=torch.bool, device=scores.device)
    masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    log_probs = torch.log_softmax(masked_scores, dim=1)
    target_probs = target_mask.to(dtype=scores.dtype, device=scores.device) / k.to(dtype=scores.dtype, device=scores.device).unsqueeze(1).clamp_min(1.0)
    return -(target_probs * log_probs.masked_fill(~valid, 0.0)).sum(dim=1).mean()


def symmetric_nce(z_soft: torch.Tensor, z_full: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = z_soft @ z_full.T / temperature
    labels = torch.arange(z_soft.shape[0], device=z_soft.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def interpolate(start: float, end: float, epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return end
    t = (epoch - 1) / (epochs - 1)
    return start * (1.0 - t) + end * t


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: SelectorTrainConfig,
    epoch: int,
    step: int,
    metrics: dict[str, float],
) -> None:
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "model_state": state,
            "config": asdict(config),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
        },
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
