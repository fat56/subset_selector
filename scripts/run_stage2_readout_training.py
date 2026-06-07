#!/usr/bin/env python3
"""Cache train500 full tokens and train the Stage 2.0 pooled readout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vggt_omega_selector.backbones.vggt_omega import VGGTOmegaIntegration  # noqa: E402
from vggt_omega_selector.readouts.training import TrainConfig, train_readout  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 2.0 train500 readout experiment.")
    parser.add_argument(
        "--manifest",
        default="docs/experiments/0003_stage2_readout_calibration/train500_manifest.json",
    )
    parser.add_argument(
        "--cache-root",
        default="caches/vggt_omega/0003_stage2_readout_calibration/train500_full16_images512",
    )
    parser.add_argument(
        "--run-dir",
        default="runs/0003_stage2_readout_calibration/train500_pooled_mlp_full16",
    )
    parser.add_argument("--checkpoint", default="512")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--cache-devices", default="cuda:0,cuda:1")
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)

    manifest_path = resolve(args.manifest)
    cache_root = resolve(args.cache_root)
    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs, records = build_full_cache_jobs(manifest, cache_root, run_dir, max_scenes=args.max_scenes)
    (run_dir / "cache_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")

    if not args.skip_cache:
        missing_jobs = [job for job in jobs if args.force_cache or not cache_ready(Path(job["output_dir"]))]
        if missing_jobs:
            run_cache_jobs_parallel(
                jobs=missing_jobs,
                run_dir=run_dir,
                checkpoint=args.checkpoint,
                image_resolution=args.image_resolution,
                mode=args.mode,
                devices=[device.strip() for device in args.cache_devices.split(",") if device.strip()],
                force=args.force_cache,
            )
        else:
            print(json.dumps({"event": "cache_skip", "reason": "all jobs already cached", "jobs": len(jobs)}), flush=True)

    validate_records(records)
    train_index = run_dir / "train_index.json"
    train_index.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")

    if args.cache_only:
        print(json.dumps({"event": "cache_only_done", "records": len(records)}, indent=2), flush=True)
        return 0

    config = TrainConfig(
        train_index=train_index,
        run_dir=run_dir,
        val_metrics_csv=resolve(
            "docs/experiments/0002_ltm30_pose_depth_validation/native_geometry/ltm30_subset_native_consistency.csv"
        ),
        val_cache_root=resolve("caches/vggt_omega/0002_ltm30_pose_depth_validation/native_geometry_images512"),
        device=args.train_device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )
    train_readout(config)
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def safe_scene_id(scene_key: str) -> str:
    return scene_key.replace("/", "__")


def build_full_cache_jobs(
    manifest: dict[str, Any],
    cache_root: Path,
    run_dir: Path,
    max_scenes: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    scenes = manifest["scenes"][:max_scenes] if max_scenes else manifest["scenes"]
    for scene in scenes:
        scene_id = safe_scene_id(scene["scene_key"])
        image_paths = [str((PROJECT_ROOT / frame["image_path"]).resolve()) for frame in scene["frames"]]
        image_list = run_dir / "image_lists" / f"{scene_id}.txt"
        image_list.parent.mkdir(parents=True, exist_ok=True)
        image_list.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        output_dir = cache_root / scene_id / "full"
        jobs.append(
            {
                "id": f"{scene_id}/full",
                "scene_id": scene_id,
                "scene_key": scene["scene_key"],
                "method": "full",
                "image_count": len(image_paths),
                "image_list": str(image_list.resolve()),
                "output_dir": str(output_dir.resolve()),
            }
        )
        records.append(
            {
                "scene_id": scene_id,
                "scene_key": scene["scene_key"],
                "dataset": scene["dataset"],
                "image_count": len(image_paths),
                "cache_dir": str(output_dir.resolve()),
                "token_path": str((output_dir / "camera_and_register_tokens.pt").resolve()),
            }
        )
    return jobs, records


def cache_ready(output_dir: Path) -> bool:
    return all(
        (output_dir / name).is_file()
        for name in ("manifest.json", "camera_and_register_tokens.pt", "register_tokens.pt")
    )


def run_cache_jobs_parallel(
    *,
    jobs: list[dict[str, Any]],
    run_dir: Path,
    checkpoint: str,
    image_resolution: int,
    mode: str,
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
        jobs_path = run_dir / f"cache_jobs_{device.replace(':', '')}.json"
        jobs_path.write_text(json.dumps(shard, indent=2) + "\n", encoding="utf-8")
        log_path = run_dir / f"cache_{device.replace(':', '')}.log"
        command = [
            str(integration.python),
            "-m",
            "vggt_omega_selector.tools.vggt_batch_cache_runner",
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
        ]
        if force:
            command.append("--force")
        print(json.dumps({"event": "cache_start", "device": device, "jobs": len(shard), "log": str(log_path)}), flush=True)
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
        print(json.dumps({"event": "cache_done", "device": device, "returncode": returncode}), flush=True)
        if returncode != 0:
            failed.append((device, returncode))
    if failed:
        raise SystemExit(f"Cache failed: {failed}")


def validate_records(records: list[dict[str, Any]]) -> None:
    missing = []
    for record in records:
        token_path = Path(record["token_path"])
        if not token_path.is_file():
            missing.append(str(token_path))
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"Missing token caches ({len(missing)}):\n{preview}")


if __name__ == "__main__":
    raise SystemExit(main())
