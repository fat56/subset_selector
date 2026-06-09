#!/usr/bin/env python3
"""Train 0005 image-only teacher/student selector against hard-native candidate labels."""

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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class CandidateRecord:
    method: str
    target_error: float
    mean_pool_register_cosine: float
    mask: torch.Tensor
    metrics: dict[str, float]


@dataclass
class SceneExample:
    scene_id: str
    scene_key: str
    dataset: str
    split: str
    feature_path: Path
    full_image_list: Path
    full_image_count: int
    candidates: list[CandidateRecord]


@dataclass
class ImageOnlyTrainConfig:
    labels_csv: Path
    cache_jobs_json: Path
    feature_cache: Path
    run_dir: Path
    train_devices: list[str]
    candidate_tag: str = "20"
    model_kind: str = "memory_candidate_set"
    seed: int = 20260609
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    limit_scenes: int | None = None
    epochs: int = 60
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    memory_slots: int = 8
    dropout: float = 0.1
    rank_weight: float = 1.0
    ce_weight: float = 0.3
    coverage_weight: float = 0.05
    min_target_gap: float = 0.02
    target_gap_scale: float = 1.0
    uniform_gate_margin: float = -1.0
    num_workers: int = 0
    eval_every_epochs: int = 1
    log_every_steps: int = 20


class MemoryCandidateSetScorer(nn.Module):
    """Score labeled candidate subsets from image-only frame features via latent scene memory."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        memory_slots: int = 8,
        dropout: float = 0.1,
        max_frames: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.memory_slots = memory_slots
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.frame_embedding = nn.Embedding(max_frames, hidden_dim)
        self.memory = nn.Parameter(torch.randn(memory_slots, hidden_dim) * 0.02)
        self.memory_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mem_attn": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                        "frame_attn": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                        "mem_norm1": nn.LayerNorm(hidden_dim),
                        "mem_norm2": nn.LayerNorm(hidden_dim),
                        "frame_norm1": nn.LayerNorm(hidden_dim),
                        "frame_norm2": nn.LayerNorm(hidden_dim),
                        "mem_ffn": ffn(hidden_dim, dropout),
                        "frame_ffn": ffn(hidden_dim, dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.candidate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 8),
            nn.Linear(hidden_dim * 4 + 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def contextualize(self, features: torch.Tensor, frame_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected features [B,N,C], got {tuple(features.shape)}")
        batch_size, frame_count, _ = features.shape
        if frame_count > self.max_frames:
            raise ValueError(f"frame_count={frame_count} exceeds max_frames={self.max_frames}")
        frame_hidden = self.projector(features.float())
        frame_ids = torch.arange(frame_count, device=features.device)
        frame_hidden = frame_hidden + self.frame_embedding(frame_ids)[None, :, :]
        memory = self.memory[None, :, :].expand(batch_size, -1, -1)
        padding_mask = ~frame_mask.to(dtype=torch.bool, device=features.device)
        for layer in self.memory_layers:
            mem_update, _ = layer["mem_attn"](
                query=layer["mem_norm1"](memory),
                key=frame_hidden,
                value=frame_hidden,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            memory = memory + mem_update
            memory = memory + layer["mem_ffn"](layer["mem_norm2"](memory))
            frame_update, _ = layer["frame_attn"](
                query=layer["frame_norm1"](frame_hidden),
                key=memory,
                value=memory,
                need_weights=False,
            )
            frame_hidden = frame_hidden + frame_update
            frame_hidden = frame_hidden + layer["frame_ffn"](layer["frame_norm2"](frame_hidden))
        frame_hidden = frame_hidden.masked_fill(padding_mask[:, :, None], 0.0)
        return frame_hidden, memory

    def forward(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> torch.Tensor:
        hidden, memory = self.contextualize(features, frame_mask)
        selected = candidate_masks.to(dtype=hidden.dtype, device=hidden.device)
        selected = selected * frame_mask.to(dtype=hidden.dtype, device=hidden.device)[:, None, :]
        denom = selected.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = torch.bmm(selected, hidden) / denom
        second = torch.bmm(selected, hidden * hidden) / denom
        std = torch.sqrt((second - mean * mean).clamp_min(0.0) + 1e-6)

        full = frame_mask.to(dtype=hidden.dtype, device=hidden.device)
        full_denom = full.sum(dim=-1, keepdim=True).clamp_min(1.0)
        full_mean = (hidden * full[:, :, None]).sum(dim=1) / full_denom
        full_mean = full_mean[:, None, :].expand_as(mean)
        memory_mean = memory.mean(dim=1)[:, None, :].expand_as(mean)
        stats = candidate_stats(selected, frame_mask)
        candidate_features = torch.cat([mean, std, full_mean, memory_mean, stats], dim=-1)
        scores = self.candidate_head(candidate_features).squeeze(-1)
        return scores.masked_fill(~candidate_valid.to(device=scores.device, dtype=torch.bool), torch.finfo(scores.dtype).min)


class MemoryFrameScorer(nn.Module):
    """Image-only per-frame scorer with the same latent scene memory."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        memory_slots: int = 8,
        dropout: float = 0.1,
        max_frames: int = 128,
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
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.context.contextualize(features, frame_mask)
        scores = self.score_head(hidden).squeeze(-1)
        padding_mask = ~frame_mask.to(dtype=torch.bool, device=features.device)
        return scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)


class ImageOnlyCandidateDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: list[SceneExample], split: str) -> None:
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
        frame_features = payload["frame_features"].float()
        image_stats = payload.get("image_stats")
        if image_stats is not None:
            frame_features = torch.cat([frame_features, image_stats.float()], dim=-1)
        candidate_masks = torch.stack([candidate.mask for candidate in example.candidates]).float()
        target_errors = torch.tensor([candidate.target_error for candidate in example.candidates], dtype=torch.float32)
        mean_pool_cosines = torch.tensor(
            [candidate.mean_pool_register_cosine for candidate in example.candidates],
            dtype=torch.float32,
        )
        return {
            "scene_id": example.scene_id,
            "scene_key": example.scene_key,
            "dataset": example.dataset,
            "features": frame_features,
            "candidate_masks": candidate_masks,
            "target_errors": target_errors,
            "mean_pool_cosines": mean_pool_cosines,
            "methods": [candidate.method for candidate in example.candidates],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train 0005 image-only fixed-K selector.")
    parser.add_argument(
        "--labels-csv",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv",
    )
    parser.add_argument(
        "--cache-jobs-json",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json",
    )
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--model-kind", choices=["memory_candidate_set", "memory_frame_score"], default="memory_candidate_set")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.3)
    parser.add_argument("--coverage-weight", type=float, default=0.05)
    parser.add_argument("--min-target-gap", type=float, default=0.02)
    parser.add_argument("--target-gap-scale", type=float, default=1.0)
    parser.add_argument(
        "--uniform-gate-margin",
        type=float,
        default=-1.0,
        help="If >=0, CE targets uniform20 unless the oracle beats uniform20 by this target-error margin.",
    )
    parser.add_argument("--train-devices", default="cuda:0,cuda:1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-steps", type=int, default=20)
    args = parser.parse_args(argv)

    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = ImageOnlyTrainConfig(
        labels_csv=resolve(args.labels_csv),
        cache_jobs_json=resolve(args.cache_jobs_json),
        feature_cache=resolve(args.feature_cache),
        run_dir=run_dir,
        train_devices=[device.strip() for device in args.train_devices.split(",") if device.strip()],
        candidate_tag=args.candidate_tag,
        model_kind=args.model_kind,
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
        rank_weight=args.rank_weight,
        ce_weight=args.ce_weight,
        coverage_weight=args.coverage_weight,
        min_target_gap=args.min_target_gap,
        target_gap_scale=args.target_gap_scale,
        uniform_gate_margin=args.uniform_gate_margin,
        num_workers=args.num_workers,
        eval_every_epochs=args.eval_every_epochs,
        log_every_steps=args.log_every_steps,
    )
    (run_dir / "train_config.json").write_text(json.dumps(asdict(config), default=str, indent=2) + "\n", encoding="utf-8")
    set_random_seeds(config.seed)
    examples = load_examples(config)
    write_dataset_summary(examples, config)
    train_selector(config, examples)
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(config: ImageOnlyTrainConfig) -> list[SceneExample]:
    jobs = json.loads(config.cache_jobs_json.read_text(encoding="utf-8"))
    jobs_by_key = {(job["scene_id"], job["method"]): job for job in jobs}
    rows_by_scene: dict[str, list[dict[str, str]]] = {}
    with config.labels_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not is_candidate_method(row["method"], config.candidate_tag):
                continue
            rows_by_scene.setdefault(row["scene_id"], []).append(row)

    raw_examples = []
    for scene_id, rows in rows_by_scene.items():
        full_job = jobs_by_key.get((scene_id, "full"))
        if full_job is None:
            continue
        full_image_list = resolve(full_job["image_list"])
        full_images = read_image_list(full_image_list)
        image_to_index = {normalise_image_path(path): index for index, path in enumerate(full_images)}
        candidates = []
        for row in sorted(rows, key=lambda item: item["method"]):
            job = jobs_by_key.get((scene_id, row["method"]))
            if job is None:
                continue
            subset_images = read_image_list(resolve(job["image_list"]))
            mask = torch.zeros(len(full_images), dtype=torch.float32)
            missing = 0
            for image_path in subset_images:
                index = image_to_index.get(normalise_image_path(image_path))
                if index is None:
                    missing += 1
                    continue
                mask[index] = 1.0
            if int(mask.sum().item()) == 0:
                continue
            metrics = {}
            for metric_name in ("pose_rotation_mean_deg", "pointmap_rmse_norm", "depth_log_rmse"):
                value = row.get(metric_name, "")
                if value != "":
                    metrics[metric_name] = float(value)
            candidates.append(
                CandidateRecord(
                    method=row["method"],
                    target_error=float(row["target_error"]),
                    mean_pool_register_cosine=float(row["mean_pool_register_cosine"]),
                    mask=mask,
                    metrics=metrics,
                )
            )
            if missing:
                print(
                    json.dumps(
                        {
                            "event": "subset_path_missing",
                            "scene_id": scene_id,
                            "method": row["method"],
                            "missing": missing,
                        }
                    ),
                    flush=True,
                )
        methods = {candidate.method for candidate in candidates}
        if f"uniform{config.candidate_tag}" not in methods:
            continue
        if len(candidates) < 3:
            continue
        feature_path = config.feature_cache / f"{scene_id}.pt"
        if not feature_path.is_file():
            continue
        raw_examples.append(
            SceneExample(
                scene_id=scene_id,
                scene_key=rows[0]["scene_key"],
                dataset=rows[0]["dataset"],
                split="unassigned",
                feature_path=feature_path,
                full_image_list=full_image_list,
                full_image_count=len(full_images),
                candidates=candidates,
            )
        )

    raw_examples = sorted(raw_examples, key=lambda example: (example.dataset, example.scene_id))
    rng = random.Random(config.seed)
    rng.shuffle(raw_examples)
    if config.limit_scenes is not None:
        raw_examples = raw_examples[: config.limit_scenes]
    examples = assign_splits(raw_examples, config)
    if not examples:
        raise ValueError(f"No examples found with feature cache at {config.feature_cache}")
    return examples


def is_candidate_method(method: str, tag: str) -> bool:
    return (
        method == f"uniform{tag}"
        or method.startswith(f"random{tag}_")
        or method.startswith(f"contiguous{tag}_")
        or method.startswith(f"uniform_jitter{tag}_")
        or method.startswith(f"convnext_kcenter{tag}_")
        or method.startswith(f"dinov2_kcenter{tag}_")
        or method.startswith(f"motion_spread{tag}_")
    )


def read_image_list(path: Path) -> list[Path]:
    return [resolve(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalise_image_path(path: Path) -> str:
    return str(path.resolve())


def assign_splits(examples: list[SceneExample], config: ImageOnlyTrainConfig) -> list[SceneExample]:
    by_dataset: dict[str, list[SceneExample]] = {}
    for example in examples:
        by_dataset.setdefault(example.dataset, []).append(example)
    rng = random.Random(config.seed)
    assigned = []
    for dataset, dataset_examples in sorted(by_dataset.items()):
        dataset_examples = list(dataset_examples)
        rng.shuffle(dataset_examples)
        count = len(dataset_examples)
        train_count = int(round(count * config.train_fraction))
        val_count = int(round(count * config.val_fraction))
        if count >= 3:
            train_count = min(max(train_count, 1), count - 2)
            val_count = min(max(val_count, 1), count - train_count - 1)
        for index, example in enumerate(dataset_examples):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            assigned.append(
                SceneExample(
                    scene_id=example.scene_id,
                    scene_key=example.scene_key,
                    dataset=example.dataset,
                    split=split,
                    feature_path=example.feature_path,
                    full_image_list=example.full_image_list,
                    full_image_count=example.full_image_count,
                    candidates=example.candidates,
                )
            )
    return sorted(assigned, key=lambda example: (example.split, example.dataset, example.scene_id))


def write_dataset_summary(examples: list[SceneExample], config: ImageOnlyTrainConfig) -> None:
    split_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    oracle_counts: dict[str, int] = {}
    backbone_counts: dict[str, int] = {}
    for example in examples:
        split_counts[example.split] = split_counts.get(example.split, 0) + 1
        dataset_counts[example.dataset] = dataset_counts.get(example.dataset, 0) + 1
        payload = torch.load(example.feature_path, map_location="cpu")
        backbone = str(payload.get("backbone", "unknown"))
        backbone_counts[backbone] = backbone_counts.get(backbone, 0) + 1
        for candidate in example.candidates:
            candidate_counts[candidate.method] = candidate_counts.get(candidate.method, 0) + 1
        best = min(example.candidates, key=lambda candidate: candidate.target_error)
        key = method_family(best.method, config.candidate_tag)
        oracle_counts[key] = oracle_counts.get(key, 0) + 1
    summary = {
        "total_scenes": len(examples),
        "split_counts": split_counts,
        "dataset_counts": dataset_counts,
        "candidate_counts": candidate_counts,
        "oracle_counts": oracle_counts,
        "backbone_counts": backbone_counts,
        "candidate_tag": config.candidate_tag,
        "feature_cache": str(config.feature_cache),
        "student_input_boundary": "image_only_no_vggt_tokens",
    }
    (config.run_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = []
    for example in examples:
        rows.append(
            {
                "scene_id": example.scene_id,
                "dataset": example.dataset,
                "split": example.split,
                "full_image_count": example.full_image_count,
                "candidate_count": len(example.candidates),
                "feature_path": str(example.feature_path),
                "full_image_list": str(example.full_image_list),
            }
        )
    with (config.run_dir / "scene_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "dataset_ready", **summary}, ensure_ascii=False), flush=True)


def collate_candidate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(items)
    max_frames = max(item["features"].shape[0] for item in items)
    max_candidates = max(item["candidate_masks"].shape[0] for item in items)
    feature_dim = items[0]["features"].shape[-1]
    features = torch.zeros((batch_size, max_frames, feature_dim), dtype=torch.float32)
    frame_mask = torch.zeros((batch_size, max_frames), dtype=torch.bool)
    candidate_masks = torch.zeros((batch_size, max_candidates, max_frames), dtype=torch.float32)
    candidate_valid = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    target_errors = torch.full((batch_size, max_candidates), 0.0, dtype=torch.float32)
    mean_pool_cosines = torch.full((batch_size, max_candidates), 0.0, dtype=torch.float32)
    uniform_indices = torch.zeros((batch_size,), dtype=torch.long)
    methods = []
    for row, item in enumerate(items):
        frame_count = item["features"].shape[0]
        candidate_count = item["candidate_masks"].shape[0]
        features[row, :frame_count] = item["features"]
        frame_mask[row, :frame_count] = True
        candidate_masks[row, :candidate_count, :frame_count] = item["candidate_masks"]
        candidate_valid[row, :candidate_count] = True
        target_errors[row, :candidate_count] = item["target_errors"]
        mean_pool_cosines[row, :candidate_count] = item["mean_pool_cosines"]
        uniform_indices[row] = find_uniform_index(item["methods"])
        methods.append(item["methods"])
    return {
        "features": features,
        "frame_mask": frame_mask,
        "candidate_masks": candidate_masks,
        "candidate_valid": candidate_valid,
        "target_errors": target_errors,
        "mean_pool_cosines": mean_pool_cosines,
        "uniform_indices": uniform_indices,
        "scene_ids": [item["scene_id"] for item in items],
        "scene_keys": [item["scene_key"] for item in items],
        "datasets": [item["dataset"] for item in items],
        "methods": methods,
    }


def find_uniform_index(methods: list[str]) -> int:
    exact_index = next((index for index, method in enumerate(methods) if method == "uniform20"), None)
    if exact_index is not None:
        return exact_index
    return next((index for index, method in enumerate(methods) if method.startswith("uniform") and "_" not in method), 0)


def train_selector(config: ImageOnlyTrainConfig, examples: list[SceneExample]) -> dict[str, Any]:
    train_ds = ImageOnlyCandidateDataset(examples, "train")
    val_ds = ImageOnlyCandidateDataset(examples, "val")
    test_ds = ImageOnlyCandidateDataset(examples, "test")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_candidate_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_candidate_batch,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_candidate_batch,
        pin_memory=True,
    )
    sample = train_ds[0]
    input_dim = int(sample["features"].shape[-1])
    max_frames = max(example.full_image_count for example in examples)
    model = build_model(config, input_dim=input_dim, max_frames=max_frames)
    device = torch.device(config.train_devices[0] if config.train_devices else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    if len(config.train_devices) > 1:
        model = nn.DataParallel(model, device_ids=[int(device_name.split(":")[-1]) for device_name in config.train_devices])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    step = 0
    best_uniform_minus_learned = -float("inf")
    best_val_error = float("inf")
    best_path = config.run_dir / "best_uniform_improvement.pt"
    started = time.time()
    initial = evaluate_selector(model, val_loader, device, config)
    print(json.dumps({"event": "eval_initial", "split": "val", **initial}, ensure_ascii=False), flush=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            scores = compute_candidate_scores(model, batch, device, config.model_kind)
            target_errors = batch["target_errors"].to(device, non_blocking=True)
            candidate_valid = batch["candidate_valid"].to(device, non_blocking=True)
            candidate_masks = batch["candidate_masks"].to(device, non_blocking=True)
            frame_mask = batch["frame_mask"].to(device, non_blocking=True)
            uniform_indices = batch["uniform_indices"].to(device, non_blocking=True)
            loss, stats = hardnative_loss(
                scores,
                target_errors,
                candidate_valid,
                candidate_masks,
                frame_mask,
                uniform_indices,
                config,
            )
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
                            "rank_loss": round(stats["rank_loss"], 6),
                            "ce_loss": round(stats["ce_loss"], 6),
                            "coverage_loss": round(stats["coverage_loss"], 6),
                            "pairwise_accuracy": round(stats["pairwise_accuracy"], 6),
                            "gate_oracle_rate": round(stats["gate_oracle_rate"], 6),
                            "steps_per_sec": round(steps_per_sec, 4),
                            "eta_sec": round(eta, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if epoch % config.eval_every_epochs == 0:
            metrics = evaluate_selector(model, val_loader, device, config)
            print(json.dumps({"event": "eval", "epoch": epoch, "split": "val", **metrics}, ensure_ascii=False), flush=True)
            improved_uniform = metrics["uniform_minus_learned_error"] > best_uniform_minus_learned
            improved_error = metrics["learned_mean_error"] < best_val_error
            if improved_uniform:
                best_uniform_minus_learned = metrics["uniform_minus_learned_error"]
                save_checkpoint(best_path, model, config, epoch, step, metrics)
            if improved_error:
                best_val_error = metrics["learned_mean_error"]
                save_checkpoint(config.run_dir / "best_val_error.pt", model, config, epoch, step, metrics)
            save_checkpoint(config.run_dir / "last.pt", model, config, epoch, step, metrics)

    if best_path.is_file():
        load_checkpoint(best_path, model, device)
    final_metrics = {
        "train": evaluate_selector(model, train_loader, device, config),
        "val": evaluate_selector(model, val_loader, device, config),
        "test": evaluate_selector(model, test_loader, device, config),
    }
    summary = {
        "epochs": config.epochs,
        "steps": step,
        "elapsed_sec": round(time.time() - started, 2),
        "best_uniform_minus_learned_error": best_uniform_minus_learned,
        "best_val_error": best_val_error,
        "best_checkpoint": str(best_path),
        "model_kind": config.model_kind,
        "student_input_boundary": "image_only_no_vggt_tokens",
        "metrics_at_best": final_metrics,
    }
    (config.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)
    return summary


def build_model(config: ImageOnlyTrainConfig, input_dim: int, max_frames: int) -> nn.Module:
    kwargs = {
        "input_dim": input_dim,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "memory_slots": config.memory_slots,
        "dropout": config.dropout,
        "max_frames": max_frames,
    }
    if config.model_kind == "memory_candidate_set":
        return MemoryCandidateSetScorer(**kwargs)
    if config.model_kind == "memory_frame_score":
        return MemoryFrameScorer(**kwargs)
    raise ValueError(f"Unsupported model_kind={config.model_kind}")


def compute_candidate_scores(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    model_kind: str,
) -> torch.Tensor:
    features = batch["features"].to(device, non_blocking=True)
    frame_mask = batch["frame_mask"].to(device, non_blocking=True)
    candidate_masks = batch["candidate_masks"].to(device, non_blocking=True)
    candidate_valid = batch["candidate_valid"].to(device, non_blocking=True)
    if model_kind == "memory_candidate_set":
        return model(features, frame_mask, candidate_masks, candidate_valid)
    frame_scores = model(features, frame_mask)
    selected = candidate_masks * frame_mask[:, None, :].to(candidate_masks.dtype)
    denom = selected.sum(dim=-1).clamp_min(1.0)
    scores = (selected * frame_scores[:, None, :]).sum(dim=-1) / denom
    return scores.masked_fill(~candidate_valid, torch.finfo(scores.dtype).min)


def hardnative_loss(
    scores: torch.Tensor,
    target_errors: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_masks: torch.Tensor,
    frame_mask: torch.Tensor,
    uniform_indices: torch.Tensor,
    config: ImageOnlyTrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = candidate_valid.to(dtype=torch.bool, device=scores.device)
    target_errors = target_errors.to(device=scores.device)
    score_i = scores.unsqueeze(2)
    score_j = scores.unsqueeze(1)
    target_i = target_errors.unsqueeze(2)
    target_j = target_errors.unsqueeze(1)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    better_i = (target_i + config.min_target_gap < target_j) & pair_valid
    score_diff = score_i - score_j
    gap_weight = ((target_j - target_i).clamp_min(0.0) / max(config.target_gap_scale, 1e-6)).clamp(0.25, 4.0)
    if better_i.any():
        pair_losses = F.softplus(-score_diff[better_i]) * gap_weight[better_i]
        rank_loss = pair_losses.mean()
        pairwise_accuracy = (score_diff[better_i] > 0).float().mean()
    else:
        rank_loss = scores.new_zeros(())
        pairwise_accuracy = scores.new_zeros(())

    masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    best_indices = target_errors.masked_fill(~valid, float("inf")).argmin(dim=1)
    if config.uniform_gate_margin >= 0.0:
        uniform_indices = uniform_indices.to(device=scores.device, dtype=torch.long)
        best_errors = target_errors.gather(1, best_indices[:, None]).squeeze(1)
        uniform_errors = target_errors.gather(1, uniform_indices[:, None]).squeeze(1)
        use_oracle = best_errors + config.uniform_gate_margin < uniform_errors
        ce_indices = torch.where(use_oracle, best_indices, uniform_indices)
        gate_oracle_rate = use_oracle.float().mean()
    else:
        ce_indices = best_indices
        gate_oracle_rate = scores.new_zeros(())
    ce_loss = F.cross_entropy(masked_scores, ce_indices)
    coverage_loss = coverage_regularizer(scores, candidate_valid, candidate_masks, frame_mask)
    loss = config.rank_weight * rank_loss + config.ce_weight * ce_loss + config.coverage_weight * coverage_loss
    return loss, {
        "rank_loss": float(rank_loss.detach().item()),
        "ce_loss": float(ce_loss.detach().item()),
        "coverage_loss": float(coverage_loss.detach().item()),
        "pairwise_accuracy": float(pairwise_accuracy.detach().item()),
        "gate_oracle_rate": float(gate_oracle_rate.detach().item()),
    }


def coverage_regularizer(
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_masks: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    valid = candidate_valid.to(dtype=torch.bool, device=scores.device)
    masks = candidate_masks.to(device=scores.device, dtype=scores.dtype)
    frame_valid = frame_mask.to(device=scores.device, dtype=scores.dtype)
    probs = torch.softmax(scores.masked_fill(~valid, torch.finfo(scores.dtype).min), dim=1)
    expected = (probs[:, :, None] * masks).sum(dim=1)
    frame_target = frame_valid * (expected.sum(dim=1, keepdim=True) / frame_valid.sum(dim=1, keepdim=True).clamp_min(1.0))
    return F.mse_loss(expected * frame_valid, frame_target)


def evaluate_selector(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    config: ImageOnlyTrainConfig,
) -> dict[str, float]:
    model.eval()
    learned_errors = []
    uniform_errors = []
    contiguous_errors = []
    random_mean_errors = []
    random_best_errors = []
    oracle_errors = []
    cosine_select_errors = []
    learned_method_counts: dict[str, int] = {}
    oracle_method_counts: dict[str, int] = {}
    learned_wins_vs_uniform = 0
    oracle_top1 = 0
    total_scenes = 0
    pair_correct = 0
    pair_total = 0
    with torch.inference_mode():
        for batch in loader:
            scores = compute_candidate_scores(model, batch, device, config.model_kind).detach().cpu()
            target_errors = batch["target_errors"]
            candidate_valid = batch["candidate_valid"]
            mean_pool_cosines = batch["mean_pool_cosines"]
            for row in range(scores.shape[0]):
                valid_indices = torch.nonzero(candidate_valid[row], as_tuple=False).flatten()
                row_scores = scores[row, valid_indices]
                row_targets = target_errors[row, valid_indices]
                row_cosines = mean_pool_cosines[row, valid_indices]
                methods = [batch["methods"][row][int(index.item())] for index in valid_indices]
                pred_local = int(torch.argmax(row_scores).item())
                oracle_local = int(torch.argmin(row_targets).item())
                cosine_local = int(torch.argmax(row_cosines).item())
                learned_method = method_family(methods[pred_local], config.candidate_tag)
                oracle_method = method_family(methods[oracle_local], config.candidate_tag)
                learned_error = float(row_targets[pred_local].item())
                oracle_error = float(row_targets[oracle_local].item())
                cosine_error = float(row_targets[cosine_local].item())
                uniform_error = method_error(methods, row_targets, f"uniform{config.candidate_tag}")
                contiguous_error = prefix_method_error(methods, row_targets, f"contiguous{config.candidate_tag}_")
                random_errors = prefix_method_errors(methods, row_targets, f"random{config.candidate_tag}_")
                if uniform_error is None or contiguous_error is None or not random_errors:
                    continue
                total_scenes += 1
                learned_errors.append(learned_error)
                uniform_errors.append(uniform_error)
                contiguous_errors.append(contiguous_error)
                random_mean_errors.append(float(sum(random_errors) / len(random_errors)))
                random_best_errors.append(float(min(random_errors)))
                oracle_errors.append(oracle_error)
                cosine_select_errors.append(cosine_error)
                learned_method_counts[learned_method] = learned_method_counts.get(learned_method, 0) + 1
                oracle_method_counts[oracle_method] = oracle_method_counts.get(oracle_method, 0) + 1
                learned_wins_vs_uniform += int(learned_error < uniform_error)
                oracle_top1 += int(pred_local == oracle_local)
                for i in range(len(row_targets)):
                    for j in range(len(row_targets)):
                        if float(row_targets[i].item()) + config.min_target_gap < float(row_targets[j].item()):
                            pair_total += 1
                            pair_correct += int(float(row_scores[i].item()) > float(row_scores[j].item()))

    learned_mean = mean(learned_errors)
    uniform_mean = mean(uniform_errors)
    random_mean = mean(random_mean_errors)
    random_best_mean = mean(random_best_errors)
    oracle_mean = mean(oracle_errors)
    cosine_mean = mean(cosine_select_errors)
    uniform_regret = uniform_mean - oracle_mean
    learned_regret = learned_mean - oracle_mean
    return {
        "scenes": float(total_scenes),
        "learned_mean_error": learned_mean,
        "uniform20_mean_error": uniform_mean,
        "contiguous20_mean_error": mean(contiguous_errors),
        "random20_mean_error": random_mean,
        "random20_best5_mean_error": random_best_mean,
        "oracle20_mean_error": oracle_mean,
        "mean_pool_cosine_select_error": cosine_mean,
        "uniform_minus_learned_error": uniform_mean - learned_mean,
        "random_mean_minus_learned_error": random_mean - learned_mean,
        "random_best5_minus_learned_error": random_best_mean - learned_mean,
        "cosine_select_minus_learned_error": cosine_mean - learned_mean,
        "uniform_regret": uniform_regret,
        "learned_regret": learned_regret,
        "regret_reduction_vs_uniform": uniform_regret - learned_regret,
        "win_rate_vs_uniform": learned_wins_vs_uniform / max(total_scenes, 1),
        "oracle_top1_rate": oracle_top1 / max(total_scenes, 1),
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
        "learned_method_counts": learned_method_counts,
        "oracle_method_counts": oracle_method_counts,
    }


def ffn(hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 4, hidden_dim),
        nn.Dropout(dropout),
    )


def candidate_stats(selected: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    frame_count = selected.shape[-1]
    denom = selected.sum(dim=-1, keepdim=True).clamp_min(1.0)
    pos = torch.linspace(0.0, 1.0, frame_count, device=selected.device, dtype=selected.dtype)
    pos = pos[None, None, :].expand_as(selected)
    pos_mean = (selected * pos).sum(dim=-1, keepdim=True) / denom
    pos_second = (selected * pos * pos).sum(dim=-1, keepdim=True) / denom
    pos_std = torch.sqrt((pos_second - pos_mean * pos_mean).clamp_min(0.0) + 1e-6)
    selected_bool = selected > 0.5
    pos_min = pos.masked_fill(~selected_bool, 1.0).amin(dim=-1, keepdim=True)
    pos_max = pos.masked_fill(~selected_bool, 0.0).amax(dim=-1, keepdim=True)
    span = (pos_max - pos_min).clamp_min(0.0)
    frame_total = frame_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(selected.dtype)[:, None, :]
    selected_fraction = selected.sum(dim=-1, keepdim=True) / frame_total
    gaps = selected[:, :, 1:] - selected[:, :, :-1]
    transitions = gaps.abs().sum(dim=-1, keepdim=True) / frame_total
    first_half = selected[:, :, : max(1, frame_count // 2)].sum(dim=-1, keepdim=True) / denom
    second_half = selected[:, :, frame_count // 2 :].sum(dim=-1, keepdim=True) / denom
    return torch.cat([selected_fraction, pos_mean, pos_std, pos_min, pos_max, span, transitions, first_half - second_half], dim=-1)


def method_error(methods: list[str], target_errors: torch.Tensor, method: str) -> float | None:
    for index, candidate_method in enumerate(methods):
        if candidate_method == method:
            return float(target_errors[index].item())
    return None


def prefix_method_error(methods: list[str], target_errors: torch.Tensor, prefix: str) -> float | None:
    for index, candidate_method in enumerate(methods):
        if candidate_method.startswith(prefix):
            return float(target_errors[index].item())
    return None


def prefix_method_errors(methods: list[str], target_errors: torch.Tensor, prefix: str) -> list[float]:
    return [float(target_errors[index].item()) for index, candidate_method in enumerate(methods) if candidate_method.startswith(prefix)]


def method_family(method: str, tag: str) -> str:
    families = (
        f"random{tag}_",
        f"contiguous{tag}_",
        f"uniform_jitter{tag}_",
        f"convnext_kcenter{tag}_",
        f"dinov2_kcenter{tag}_",
        f"motion_spread{tag}_",
    )
    for prefix in families:
        if method.startswith(prefix):
            return prefix.removesuffix("_")
    return method


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: ImageOnlyTrainConfig,
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
