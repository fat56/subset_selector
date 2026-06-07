"""Batch VGGT-OMEGA cache runner.

This module is launched inside the VGGT-OMEGA Python environment. It loads the
model once, then processes multiple image-list cache jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache VGGT-OMEGA outputs for multiple image lists.")
    parser.add_argument("--vggt-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jobs-json", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-pose", action="store_true")
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--enable-alignment", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    vggt_root = Path(args.vggt_root).resolve()
    sys.path.insert(0, str(vggt_root))

    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("jobs-json must contain a non-empty list")

    device = resolve_device(torch, args.device)
    checkpoint_path = Path(args.checkpoint).resolve()

    model = VGGTOmega(
        enable_camera=args.include_pose,
        enable_depth=args.include_depth,
        enable_alignment=args.enable_alignment,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    load_result = model.load_state_dict(state_dict, strict=args.strict_load)
    del state_dict
    model.eval()
    patch_token_start = int(getattr(model.aggregator, "patch_token_start"))

    statuses: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        job_id = str(job.get("id") or f"job_{index:04d}")
        output_dir = Path(job["output_dir"]).resolve()
        image_paths = collect_image_paths(job)
        if not image_paths:
            statuses.append({"id": job_id, "status": "failed", "error": "no images"})
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        embedding_path = output_dir / "register_mean_embedding.json"
        required_outputs = [output_dir / "register_tokens.pt", embedding_path]
        if args.include_pose:
            required_outputs.append(output_dir / "pose_enc.pt")
        if args.include_depth:
            required_outputs.extend([output_dir / "depth.pt", output_dir / "depth_conf.pt"])
        if args.enable_alignment:
            required_outputs.extend(
                [output_dir / "text_alignment_embedding.pt", output_dir / "text_alignment_token.pt"]
            )
        if args.include_images:
            required_outputs.append(output_dir / "images.pt")

        if not args.force and manifest_path.exists() and all(path.exists() for path in required_outputs):
            statuses.append({"id": job_id, "status": "skipped", "output_dir": str(output_dir)})
            continue

        print(json.dumps({"event": "start", "index": index, "total": len(jobs), "id": job_id, "images": len(image_paths)}))
        try:
            images = load_and_preprocess_images(
                image_paths,
                mode=args.mode,
                image_resolution=args.image_resolution,
            ).to(device)

            with torch.inference_mode():
                predictions = model(images)

            outputs: dict[str, dict[str, Any]] = {}
            camera_and_register = predictions["camera_and_register_tokens"].detach().cpu()
            save_tensor(torch, output_dir, outputs, "camera_and_register_tokens", camera_and_register)
            save_tensor(torch, output_dir, outputs, "camera_tokens", camera_and_register[:, :, :1])
            register_tokens = camera_and_register[:, :, 1:]
            save_tensor(torch, output_dir, outputs, "register_tokens", register_tokens)
            save_register_mean_embedding(output_dir, register_tokens)

            optional_keys = []
            if args.include_pose:
                optional_keys.append("pose_enc")
            if args.include_depth:
                optional_keys.extend(["depth", "depth_conf"])
            if args.enable_alignment:
                optional_keys.extend(["text_alignment_embedding", "text_alignment_token"])
            if args.include_images:
                optional_keys.append("images")

            for key in optional_keys:
                value = predictions.get(key)
                if torch.is_tensor(value):
                    save_tensor(torch, output_dir, outputs, key, value.detach().cpu())

            manifest = {
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000"),
                "job": job,
                "vggt_root": str(vggt_root),
                "checkpoint": str(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
                "image_resolution": args.image_resolution,
                "mode": args.mode,
                "device": str(device),
                "images": image_paths,
                "input_tensor_shape": list(images.shape),
                "patch_token_start": patch_token_start,
                "num_register_tokens": patch_token_start - 1,
                "outputs": outputs,
                "register_mean_embedding": "register_mean_embedding.json",
                "load_state_dict": {
                    "strict": args.strict_load,
                    "missing_keys": list(getattr(load_result, "missing_keys", [])),
                    "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
                },
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            statuses.append({"id": job_id, "status": "done", "output_dir": str(output_dir)})
            print(json.dumps({"event": "done", "index": index, "total": len(jobs), "id": job_id}))
        except Exception as exc:  # noqa: BLE001 - keep long batch jobs inspectable.
            statuses.append({"id": job_id, "status": "failed", "error": repr(exc), "output_dir": str(output_dir)})
            print(json.dumps({"event": "failed", "index": index, "total": len(jobs), "id": job_id, "error": repr(exc)}))
        finally:
            try:
                del images
            except UnboundLocalError:
                pass
            try:
                del predictions
            except UnboundLocalError:
                pass
            try:
                del camera_and_register
            except UnboundLocalError:
                pass
            try:
                del register_tokens
            except UnboundLocalError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    failed = [status for status in statuses if status["status"] == "failed"]
    print(json.dumps({"ok": not failed, "jobs": len(statuses), "failed": len(failed)}, indent=2))
    return 1 if failed else 0


def collect_image_paths(job: dict[str, Any]) -> list[str]:
    if job.get("image_list"):
        list_path = Path(job["image_list"])
        return [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [str(path) for path in job.get("images", [])]


def resolve_device(torch_module: Any, device_arg: str) -> str:
    if device_arg == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        raise SystemExit("VGGT-OMEGA inference requires CUDA with the released forward path.")
    if device_arg == "cpu":
        raise SystemExit("CPU execution is not supported by the released VGGT-OMEGA forward path.")
    return device_arg


def save_tensor(torch_module: Any, output_dir: Path, outputs: dict[str, dict[str, Any]], key: str, tensor: Any) -> None:
    path = output_dir / f"{key}.pt"
    torch_module.save(tensor, path)
    outputs[key] = {
        "path": path.name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def save_register_mean_embedding(output_dir: Path, register_tokens: Any) -> None:
    pooled = register_tokens.float().mean(dim=(0, 1, 2))
    payload = {
        "pooling": "mean_over_batch_frames_and_register_tokens",
        "source_tensor": "register_tokens.pt",
        "source_shape": list(register_tokens.shape),
        "embedding_dim": int(pooled.numel()),
        "embedding": [float(value) for value in pooled.tolist()],
    }
    (output_dir / "register_mean_embedding.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
