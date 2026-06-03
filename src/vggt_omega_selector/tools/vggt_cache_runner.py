"""Run VGGT-OMEGA inference and save a lightweight tensor cache.

This module is meant to be launched by the VGGT-OMEGA Python environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache VGGT-OMEGA predictions for a list of images.")
    parser.add_argument("--vggt-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image", action="append", default=[], dest="images")
    parser.add_argument("--image-list", default=None)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-pose", action="store_true")
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--enable-alignment", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    args = parser.parse_args(argv)

    vggt_root = Path(args.vggt_root).resolve()
    sys.path.insert(0, str(vggt_root))

    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    image_paths = collect_image_paths(args.images, args.image_list)
    if not image_paths:
        raise SystemExit("No images provided. Use --image or --image-list.")

    device = resolve_device(torch, args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = VGGTOmega(
        enable_camera=args.include_pose,
        enable_depth=args.include_depth,
        enable_alignment=args.enable_alignment,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    load_result = model.load_state_dict(state_dict, strict=args.strict_load)
    model.eval()

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
    save_tensor(torch, output_dir, outputs, "register_tokens", camera_and_register[:, :, 1:])

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

    patch_token_start = int(getattr(model.aggregator, "patch_token_start"))
    manifest = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000"),
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
        "load_state_dict": {
            "strict": args.strict_load,
            "missing_keys": list(getattr(load_result, "missing_keys", [])),
            "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "outputs": sorted(outputs)}, indent=2))
    return 0


def collect_image_paths(images: list[str], image_list: str | None) -> list[str]:
    paths = list(images)
    if image_list:
        list_path = Path(image_list)
        for line in list_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                paths.append(stripped)
    return paths


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


if __name__ == "__main__":
    raise SystemExit(main())

