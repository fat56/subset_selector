#!/usr/bin/env python3
"""Cache image-only features for 0005 teacher/student selector experiments."""

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
from PIL import Image
from PIL import ImageFile
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class FeatureScene:
    scene_id: str
    scene_key: str
    dataset: str
    image_list: Path
    output_path: Path
    image_count: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare 0005 image-only selector feature cache.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional scene manifest with full frames. If provided, labels/cache-jobs inputs are not used.",
    )
    parser.add_argument(
        "--labels-csv",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv",
    )
    parser.add_argument(
        "--cache-jobs-json",
        default="runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--backbone",
        choices=["convnext_tiny", "dinov2_vits14", "dinov2_vits14_patch_summary", "mobilenet_v3_large"],
        default="convnext_tiny",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-tag", default="20")
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dinov2-hub-dir", default="/home/m/.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--temporal-stats", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard_count")

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        scenes = load_feature_scenes_from_manifest(
            manifest_path=resolve(args.manifest),
            out_dir=out_dir,
            limit_scenes=args.limit_scenes,
            seed=args.seed,
        )
    else:
        scenes = load_feature_scenes(
            labels_csv=resolve(args.labels_csv),
            cache_jobs_json=resolve(args.cache_jobs_json),
            out_dir=out_dir,
            candidate_tag=args.candidate_tag,
            limit_scenes=args.limit_scenes,
            seed=args.seed,
        )
    scenes = [scene for index, scene in enumerate(scenes) if index % args.shard_count == args.shard_index]
    metadata = {
        "backbone": args.backbone,
        "manifest": str(resolve(args.manifest)) if args.manifest else None,
        "labels_csv": str(resolve(args.labels_csv)),
        "cache_jobs_json": str(resolve(args.cache_jobs_json)),
        "out_dir": str(out_dir),
        "candidate_tag": args.candidate_tag,
        "limit_scenes": args.limit_scenes,
        "seed": args.seed,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "total_shard_scenes": len(scenes),
        "temporal_stats": bool(args.temporal_stats),
        "student_input_boundary": "image_only_no_vggt_tokens",
    }
    (out_dir / f"prepare_metadata_shard{args.shard_index:02d}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "feature_prepare_start", **metadata}, ensure_ascii=False), flush=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    extractor = build_extractor(args.backbone, device=device, dinov2_hub_dir=Path(args.dinov2_hub_dir))
    cache_features(
        scenes=scenes,
        extractor=extractor,
        backbone=args.backbone,
        device=device,
        batch_size=args.batch_size,
        temporal_stats=args.temporal_stats,
        force=args.force,
    )
    return 0


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_feature_scenes(
    labels_csv: Path,
    cache_jobs_json: Path,
    out_dir: Path,
    candidate_tag: str,
    limit_scenes: int | None,
    seed: int,
) -> list[FeatureScene]:
    wanted_scene_ids = set()
    methods_by_scene: dict[str, set[str]] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if not is_candidate_method(method, candidate_tag):
                continue
            wanted_scene_ids.add(row["scene_id"])
            methods_by_scene.setdefault(row["scene_id"], set()).add(method)

    jobs = json.loads(cache_jobs_json.read_text(encoding="utf-8"))
    scenes = []
    for job in jobs:
        if job.get("method") != "full":
            continue
        scene_id = job["scene_id"]
        if scene_id not in wanted_scene_ids:
            continue
        if f"uniform{candidate_tag}" not in methods_by_scene.get(scene_id, set()):
            continue
        image_list = resolve(job["image_list"])
        image_count = len(read_image_list(image_list))
        scenes.append(
            FeatureScene(
                scene_id=scene_id,
                scene_key=job["scene_key"],
                dataset=job["dataset"],
                image_list=image_list,
                output_path=out_dir / f"{scene_id}.pt",
                image_count=image_count,
            )
        )

    scenes = sorted(scenes, key=lambda scene: (scene.dataset, scene.scene_id))
    rng = random.Random(seed)
    rng.shuffle(scenes)
    if limit_scenes is not None:
        scenes = scenes[:limit_scenes]
    return scenes


def load_feature_scenes_from_manifest(
    manifest_path: Path,
    out_dir: Path,
    limit_scenes: int | None,
    seed: int,
) -> list[FeatureScene]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = []
    image_list_root = out_dir / "image_lists"
    for scene in manifest["scenes"]:
        scene_id = scene.get("scene_id") or scene["scene_key"].replace("/", "__")
        image_paths = [resolve(frame["image_path"]) for frame in scene["frames"]]
        image_list = image_list_root / f"{scene_id}.txt"
        image_list.parent.mkdir(parents=True, exist_ok=True)
        image_list.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")
        scenes.append(
            FeatureScene(
                scene_id=scene_id,
                scene_key=scene["scene_key"],
                dataset=scene["dataset"],
                image_list=image_list,
                output_path=out_dir / f"{scene_id}.pt",
                image_count=len(image_paths),
            )
        )

    scenes = sorted(scenes, key=lambda scene: (scene.dataset, scene.scene_id))
    rng = random.Random(seed)
    rng.shuffle(scenes)
    if limit_scenes is not None:
        scenes = scenes[:limit_scenes]
    return scenes


def is_candidate_method(method: str, tag: str) -> bool:
    return (
        method == f"uniform{tag}"
        or method.startswith(f"random{tag}_")
        or method.startswith(f"contiguous{tag}_")
        or method.startswith(f"uniform_jitter{tag}_")
        or method.startswith(f"swapgain{tag}_")
        or method.startswith(f"convnext_kcenter{tag}_")
        or method.startswith(f"dinov2_kcenter{tag}_")
        or method.startswith(f"motion_spread{tag}_")
    )


def read_image_list(path: Path) -> list[Path]:
    return [resolve(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ImageFeatureExtractor:
    def __init__(self, backbone: str, model: nn.Module, transform: Any, device: torch.device) -> None:
        self.backbone = backbone
        self.model = model
        self.transform = transform
        self.device = device

    @torch.inference_mode()
    def encode(self, image_paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = []
        stats = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            stats.append(image_stats(image))
            tensors.append(self.transform(image))
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
        if self.backbone == "dinov2_vits14_patch_summary":
            output = self.model.forward_features(batch)
            features = dinov2_patch_summary(
                cls_token=output["x_norm_clstoken"],
                patch_tokens=output["x_norm_patchtokens"],
            )
        else:
            features = self.model(batch)
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                features = features["x_norm_clstoken"]
            else:
                raise TypeError(f"Unsupported feature dict keys from {self.backbone}: {sorted(features)}")
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features.detach().cpu().float(), torch.stack(stats, dim=0).float()


def build_extractor(backbone: str, device: torch.device, dinov2_hub_dir: Path) -> ImageFeatureExtractor:
    if backbone in {"convnext_tiny", "mobilenet_v3_large"}:
        import torchvision.models as models

        if backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            model = models.convnext_tiny(weights=weights)
            model.classifier = nn.Identity()
        else:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            model = models.mobilenet_v3_large(weights=weights)
            model.classifier = nn.Identity()
        transform = weights.transforms()
    elif backbone in {"dinov2_vits14", "dinov2_vits14_patch_summary"}:
        import torchvision.transforms as T

        if not dinov2_hub_dir.exists():
            raise FileNotFoundError(f"DINOv2 hub dir not found: {dinov2_hub_dir}")
        model = torch.hub.load(str(dinov2_hub_dir), "dinov2_vits14", source="local", pretrained=True)
        transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    else:
        raise ValueError(f"Unsupported backbone={backbone}")
    model.eval().to(device)
    return ImageFeatureExtractor(backbone=backbone, model=model, transform=transform, device=device)


def dinov2_patch_summary(cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected DINOv2 patch tokens [B,P,C], got {tuple(patch_tokens.shape)}")
    patch_mean = patch_tokens.mean(dim=1)
    patch_std = patch_tokens.std(dim=1, unbiased=False)
    patch_count = patch_tokens.shape[1]
    grid_size = int(patch_count**0.5)
    if grid_size * grid_size != patch_count:
        return torch.cat([cls_token, patch_mean, patch_std], dim=-1)

    grid = patch_tokens.reshape(patch_tokens.shape[0], grid_size, grid_size, patch_tokens.shape[-1])
    half = grid_size // 2
    top_bottom = grid[:, :half].mean(dim=(1, 2)) - grid[:, half:].mean(dim=(1, 2))
    left_right = grid[:, :, :half].mean(dim=(1, 2)) - grid[:, :, half:].mean(dim=(1, 2))
    margin = max(grid_size // 4, 1)
    center = grid[:, margin:-margin, margin:-margin].mean(dim=(1, 2))
    border_mask = torch.ones((grid_size, grid_size), dtype=torch.bool, device=grid.device)
    border_mask[margin:-margin, margin:-margin] = False
    border = grid[:, border_mask].mean(dim=1)
    center_border = center - border
    return torch.cat([cls_token, patch_mean, patch_std, top_bottom, left_right, center_border], dim=-1)


def image_stats(image: Image.Image) -> torch.Tensor:
    import torchvision.transforms.functional as TF

    tensor = TF.resize(image, [64, 64], interpolation=TF.InterpolationMode.BILINEAR)
    values = TF.to_tensor(tensor)
    rgb_mean = values.mean(dim=(1, 2))
    rgb_std = values.std(dim=(1, 2), unbiased=False)
    gray = values.mean(dim=0)
    gray_mean = gray.mean().view(1)
    gray_std = gray.std(unbiased=False).view(1)
    grad_x = (gray[:, 1:] - gray[:, :-1]).abs().mean().view(1)
    grad_y = (gray[1:, :] - gray[:-1, :]).abs().mean().view(1)
    return torch.cat([rgb_mean, rgb_std, gray_mean, gray_std, grad_x, grad_y], dim=0)


def temporal_feature_stats(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected features [N,C], got {tuple(features.shape)}")
    if features.shape[0] == 0:
        return torch.zeros((0, 6), dtype=torch.float32)
    values = features.float()
    normalised = torch.nn.functional.normalize(values, dim=-1)
    prev = torch.cat([normalised[:1], normalised[:-1]], dim=0)
    next_ = torch.cat([normalised[1:], normalised[-1:]], dim=0)
    prev_cos = (normalised * prev).sum(dim=-1, keepdim=True)
    next_cos = (normalised * next_).sum(dim=-1, keepdim=True)

    prev_values = torch.cat([values[:1], values[:-1]], dim=0)
    next_values = torch.cat([values[1:], values[-1:]], dim=0)
    scale = max(float(values.shape[-1]) ** 0.5, 1.0)
    prev_l2 = (values - prev_values).norm(dim=-1, keepdim=True) / scale
    next_l2 = (values - next_values).norm(dim=-1, keepdim=True) / scale
    centred = values - values.mean(dim=0, keepdim=True)
    scene_l2 = centred.norm(dim=-1, keepdim=True) / scale
    frame_pos = torch.linspace(0.0, 1.0, steps=values.shape[0], dtype=torch.float32).unsqueeze(-1)
    return torch.cat([prev_cos, next_cos, prev_l2, next_l2, scene_l2, frame_pos], dim=-1)


def cache_features(
    scenes: list[FeatureScene],
    extractor: ImageFeatureExtractor,
    backbone: str,
    device: torch.device,
    batch_size: int,
    temporal_stats: bool,
    force: bool,
) -> None:
    started = time.time()
    done = 0
    skipped = 0
    for scene_index, scene in enumerate(scenes, start=1):
        if scene.output_path.is_file() and not force:
            skipped += 1
            done += 1
            continue
        image_paths = read_image_list(scene.image_list)
        frame_features = []
        frame_stats = []
        for start in range(0, len(image_paths), batch_size):
            features, stats = extractor.encode(image_paths[start : start + batch_size])
            frame_features.append(features)
            frame_stats.append(stats)
        features_tensor = torch.cat(frame_features, dim=0)
        stats_tensor = torch.cat(frame_stats, dim=0)
        if temporal_stats:
            stats_tensor = torch.cat([stats_tensor, temporal_feature_stats(features_tensor)], dim=-1)
        if features_tensor.shape[0] != scene.image_count:
            raise ValueError(f"Feature count mismatch for {scene.scene_id}: {features_tensor.shape[0]} vs {scene.image_count}")
        scene.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = scene.output_path.with_suffix(f".tmp.{scene.output_path.name}.{int(time.time() * 1000)}")
        torch.save(
            {
                "scene_id": scene.scene_id,
                "scene_key": scene.scene_key,
                "dataset": scene.dataset,
                "backbone": backbone,
                "feature_kind": "image_only_no_vggt_tokens",
                "image_list": [str(path) for path in image_paths],
                "frame_features": features_tensor.to(dtype=torch.float16),
                "image_stats": stats_tensor.to(dtype=torch.float16),
                "feature_dim": int(features_tensor.shape[-1]),
                "stats_dim": int(stats_tensor.shape[-1]),
            },
            tmp_path,
        )
        tmp_path.replace(scene.output_path)
        done += 1
        elapsed = time.time() - started
        scenes_per_sec = done / max(elapsed, 1e-6)
        eta = (len(scenes) - done) / max(scenes_per_sec, 1e-6)
        print(
            json.dumps(
                {
                    "event": "feature_cached",
                    "scene_index": scene_index,
                    "done": done,
                    "total": len(scenes),
                    "skipped": skipped,
                    "scene_id": scene.scene_id,
                    "frames": scene.image_count,
                    "feature_dim": int(features_tensor.shape[-1]),
                    "device": str(device),
                    "elapsed_sec": round(elapsed, 1),
                    "eta_sec": round(eta, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "event": "feature_prepare_done",
                "done": done,
                "skipped": skipped,
                "total": len(scenes),
                "elapsed_sec": round(elapsed, 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
