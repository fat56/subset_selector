"""Cache compact per-image selector features from VGGT-OMEGA tokens."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache compact selector features for multiple image-list jobs.")
    parser.add_argument("--vggt-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jobs-json", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    vggt_root = Path(args.vggt_root).resolve()
    sys.path.insert(0, str(vggt_root))

    import torch
    from torch.nn import functional as F
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("jobs-json must contain a non-empty list")

    device = resolve_device(torch, args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    feature_dtype = torch.float16 if args.feature_dtype == "float16" else torch.float32

    model = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False).to(device)
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
        feature_path = output_dir / "selector_features.pt"
        manifest_path = output_dir / "manifest.json"
        if not args.force and feature_path.exists() and manifest_path.exists():
            statuses.append({"id": job_id, "status": "skipped", "output_dir": str(output_dir)})
            continue

        print(json.dumps({"event": "start", "index": index, "total": len(jobs), "id": job_id, "images": len(image_paths)}), flush=True)
        try:
            images = load_and_preprocess_images(
                image_paths,
                mode=args.mode,
                image_resolution=args.image_resolution,
            ).to(device)
            with torch.inference_mode():
                predictions = model(images)

            camera_and_register = predictions["camera_and_register_tokens"].detach().cpu()
            if camera_and_register.ndim != 4 or camera_and_register.shape[0] != 1:
                raise RuntimeError(f"Unexpected camera/register token shape: {tuple(camera_and_register.shape)}")

            tokens = camera_and_register[0].float()
            camera = tokens[:, 0, :]
            register = tokens[:, 1:, :]
            register_mean = register.mean(dim=1)
            register_max = register.amax(dim=1)
            register_std = register.std(dim=1, unbiased=False)
            if register.shape[0] == 1:
                frame_pos = torch.zeros((1, 1), dtype=torch.float32)
            else:
                frame_pos = torch.linspace(0.0, 1.0, register.shape[0], dtype=torch.float32).unsqueeze(-1)
            frame_features = torch.cat([camera, register_mean, register_max, register_std, frame_pos], dim=-1)
            full_embedding = F.normalize(register.mean(dim=(0, 1)), dim=0)

            payload = {
                "frame_features": frame_features.to(feature_dtype).contiguous(),
                "register_mean": register_mean.to(feature_dtype).contiguous(),
                "full_embedding": full_embedding.float().contiguous(),
                "frame_ids": list(job.get("frame_ids") or [Path(path).stem for path in image_paths]),
                "image_paths": image_paths,
                "scene_id": job.get("scene_id"),
                "scene_key": job.get("scene_key"),
                "dataset": job.get("dataset"),
            }
            torch.save(payload, feature_path)

            manifest = {
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000"),
                "job": job,
                "vggt_root": str(vggt_root),
                "checkpoint": str(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
                "image_resolution": args.image_resolution,
                "mode": args.mode,
                "device": str(device),
                "feature_dtype": args.feature_dtype,
                "images": image_paths,
                "input_tensor_shape": list(images.shape),
                "patch_token_start": patch_token_start,
                "num_register_tokens": patch_token_start - 1,
                "outputs": {
                    "selector_features": {
                        "path": feature_path.name,
                        "frame_features_shape": list(frame_features.shape),
                        "register_mean_shape": list(register_mean.shape),
                        "full_embedding_shape": list(full_embedding.shape),
                    }
                },
                "load_state_dict": {
                    "strict": args.strict_load,
                    "missing_keys": list(getattr(load_result, "missing_keys", [])),
                    "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
                },
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            statuses.append({"id": job_id, "status": "done", "output_dir": str(output_dir)})
            print(json.dumps({"event": "done", "index": index, "total": len(jobs), "id": job_id}), flush=True)
        except Exception as exc:  # noqa: BLE001
            statuses.append({"id": job_id, "status": "failed", "error": repr(exc), "output_dir": str(output_dir)})
            print(json.dumps({"event": "failed", "index": index, "total": len(jobs), "id": job_id, "error": repr(exc)}), flush=True)
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
                del tokens
            except UnboundLocalError:
                pass
            try:
                del camera
            except UnboundLocalError:
                pass
            try:
                del register
            except UnboundLocalError:
                pass
            try:
                del register_mean
            except UnboundLocalError:
                pass
            try:
                del register_max
            except UnboundLocalError:
                pass
            try:
                del register_std
            except UnboundLocalError:
                pass
            try:
                del frame_pos
            except UnboundLocalError:
                pass
            try:
                del frame_features
            except UnboundLocalError:
                pass
            try:
                del full_embedding
            except UnboundLocalError:
                pass
            try:
                del payload
            except UnboundLocalError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    failed = [status for status in statuses if status["status"] == "failed"]
    print(json.dumps({"ok": not failed, "jobs": len(statuses), "failed": len(failed)}, indent=2), flush=True)
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


if __name__ == "__main__":
    raise SystemExit(main())
