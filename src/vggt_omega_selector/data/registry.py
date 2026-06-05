"""Dataset registry helpers for experiment scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vggt_omega_selector.project import find_project_root


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class SceneRecord:
    """A registered scene that can be prepared for Stage 1."""

    dataset_id: str
    scene_id: str
    root: Path
    image_dir: str = "images"
    sparse_dir: str = "sparse/0"
    image_glob: str = "*"
    features: dict[str, Path] = field(default_factory=dict)
    notes: str = ""

    @property
    def images_path(self) -> Path:
        return self.root / self.image_dir

    @property
    def sparse_path(self) -> Path:
        return self.root / self.sparse_dir


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {source}")
    return payload


def load_scene_records(registry_path: str | Path, *, project_root: Path | None = None) -> list[SceneRecord]:
    """Load scene records from ``data/datasets.yaml``."""

    project = project_root or find_project_root()
    source = project / registry_path if not Path(registry_path).is_absolute() else Path(registry_path)
    payload = load_yaml(source)
    datasets = payload.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("datasets must be a list")

    scenes: list[SceneRecord] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("each dataset entry must be a mapping")
        dataset_id = str(dataset["id"])
        dataset_root = resolve_path(dataset.get("root", "."), project)
        dataset_image_dir = str(dataset.get("image_dir", "images"))
        dataset_sparse_dir = str(dataset.get("sparse_dir", "sparse/0"))
        dataset_image_glob = str(dataset.get("image_glob", "*"))
        dataset_features = resolve_feature_paths(dataset.get("features", {}), dataset_root, project)

        for scene in dataset.get("scenes", []):
            if isinstance(scene, str):
                scene_id = scene
                scene_root = dataset_root / scene
                image_dir = dataset_image_dir
                sparse_dir = dataset_sparse_dir
                image_glob = dataset_image_glob
                features = dataset_features
                notes = ""
            elif isinstance(scene, dict):
                scene_id = str(scene["id"])
                scene_root = resolve_path(scene.get("root", dataset_root / scene_id), project, base=dataset_root)
                image_dir = str(scene.get("image_dir", dataset_image_dir))
                sparse_dir = str(scene.get("sparse_dir", dataset_sparse_dir))
                image_glob = str(scene.get("image_glob", dataset_image_glob))
                features = {
                    **dataset_features,
                    **resolve_feature_paths(scene.get("features", {}), scene_root, project),
                }
                notes = str(scene.get("notes", ""))
            else:
                raise ValueError("scene entries must be strings or mappings")

            scenes.append(
                SceneRecord(
                    dataset_id=dataset_id,
                    scene_id=scene_id,
                    root=scene_root,
                    image_dir=image_dir,
                    sparse_dir=sparse_dir,
                    image_glob=image_glob,
                    features=features,
                    notes=notes,
                )
            )
    return scenes


def list_scene_images(scene: SceneRecord) -> list[Path]:
    """Return scene image paths in deterministic dataset order."""

    image_root = scene.images_path
    if not image_root.exists():
        return []
    images = [
        path
        for path in image_root.glob(scene.image_glob)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(images, key=lambda path: path.relative_to(image_root).as_posix())


def resolve_feature_paths(payload: Any, base: Path, project: Path) -> dict[str, Path]:
    if not payload:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("features must be a mapping from method id to path")
    return {str(key): resolve_path(value, project, base=base) for key, value in payload.items()}


def resolve_path(value: Any, project: Path, *, base: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if base is not None:
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
        if path.parts and path.parts[0] not in {"caches", "configs", "data", "docs", "external", "runs"}:
            return candidate.resolve()
    return (project / path).resolve()
