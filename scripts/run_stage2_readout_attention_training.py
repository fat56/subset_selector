#!/usr/bin/env python3
"""Train the Stage 2.0 attention multi-metric readout from hard labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vggt_omega_selector.readouts.training import (  # noqa: E402
    AttentionHardLabelTrainConfig,
    PRIMARY_VAL_METRICS,
    train_attention_multimetric_readout,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run attention multi-metric readout training.")
    parser.add_argument(
        "--labels-csv",
        default=(
            "runs/0003_stage2_readout_calibration/"
            "hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv"
        ),
    )
    parser.add_argument(
        "--run-dir",
        default="runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric",
    )
    parser.add_argument("--train-devices", default="cuda:0,cuda:1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pairs-per-scene-metric", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args(argv)

    config = AttentionHardLabelTrainConfig(
        labels_csv=resolve(args.labels_csv),
        run_dir=resolve(args.run_dir),
        val_metrics_csv=resolve(
            "docs/experiments/0002_ltm30_pose_depth_validation/native_geometry/ltm30_subset_native_consistency.csv"
        ),
        val_cache_root=resolve("caches/vggt_omega/0002_ltm30_pose_depth_validation/native_geometry_images512"),
        devices=tuple(device.strip() for device in args.train_devices.split(",") if device.strip()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        pairs_per_scene_metric=args.pairs_per_scene_metric,
        metrics=PRIMARY_VAL_METRICS,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "config.json").write_text(
        json.dumps(
            {
                "labels_csv": str(config.labels_csv),
                "devices": list(config.devices),
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "num_workers": config.num_workers,
                "pairs_per_scene_metric": config.pairs_per_scene_metric,
                "metrics": list(config.metrics),
                "hidden_dim": config.hidden_dim,
                "output_dim": config.output_dim,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "dropout": config.dropout,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    train_attention_multimetric_readout(config)
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
