"""Stage 1 preparation workflow."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vggt_omega_selector.baselines.ratio import (
    FEATURE_K_CENTER_METHODS,
    REGISTER_K_CENTER_METHODS,
    K_CENTER_METHODS,
    load_feature_vectors,
    select_ratio_indices,
)
from vggt_omega_selector.data.colmap_subset import materialize_colmap_subset
from vggt_omega_selector.data.registry import SceneRecord, list_scene_images, load_scene_records, load_yaml
from vggt_omega_selector.evaluation.fastgs import FastGSIntegration
from vggt_omega_selector.project import find_project_root, relative_to_root


DEFAULT_STAGE1_CONFIG = "configs/experiments/0001_stage1_register_quality_gate.yaml"


@dataclass(frozen=True)
class PreparedStage1Run:
    """A single prepared scene/method subset."""

    method: str
    dataset_id: str
    scene_id: str
    status: str
    total_images: int
    selected_count: int
    requested_ratio: float
    actual_ratio: float
    work_dir: str
    selected_indices: str | None
    subset_source: str | None
    fastgs_command: str | None
    notes: str


def prepare_stage1(
    *,
    config_path: str | Path = DEFAULT_STAGE1_CONFIG,
    dataset_id: str | None = None,
    scene_ids: list[str] | None = None,
    methods: list[str] | None = None,
    ratio: float | None = None,
    output_root: str | Path | None = None,
    seed: int = 13,
    overwrite: bool = False,
    materialize: bool = True,
    fastgs_root: str | Path | None = None,
    fastgs_python: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare ratio-based Stage 1 subset runs without launching FastGS."""

    project = find_project_root()
    config_source = project / config_path if not Path(config_path).is_absolute() else Path(config_path)
    config = load_yaml(config_source)
    experiment = config.get("experiment", {})
    experiment_id = str(experiment.get("id", "stage1"))
    selected_ratio = ratio if ratio is not None else first_ratio(config)
    selected_methods = methods if methods else list(config.get("selection_methods", []))
    registry_path = config.get("datasets", {}).get("registry", "data/datasets.yaml")
    scenes = filter_scenes(load_scene_records(registry_path, project_root=project), dataset_id, scene_ids)
    fastgs = FastGSIntegration.discover(root=fastgs_root, python=fastgs_python, project_root=project)
    fastgs_options = config.get("evaluation", {}).get("fastgs", {})
    root = Path(output_root) if output_root else project / "runs" / experiment_id / "prepared"
    if not root.is_absolute():
        root = project / root

    prepared: list[PreparedStage1Run] = []
    warnings: list[str] = []
    if not scenes:
        warnings.append("No matching scenes are registered yet.")

    for scene in scenes:
        image_paths = list_scene_images(scene)
        image_names = [path.relative_to(scene.images_path).as_posix() for path in image_paths]
        if not image_paths:
            warnings.append(f"{scene.dataset_id}/{scene.scene_id}: no images found at {scene.images_path}")
            continue

        for method in selected_methods:
            work_dir = root / safe_id(scene.dataset_id) / safe_id(scene.scene_id) / safe_id(method) / ratio_slug(selected_ratio)
            try:
                feature_vectors = load_method_features(scene, method, image_names)
                indices = select_ratio_indices(
                    method,
                    len(image_paths),
                    selected_ratio,
                    seed=seed,
                    feature_vectors=feature_vectors,
                )
            except ValueError as exc:
                prepared.append(
                    write_pending_manifest(
                        work_dir=work_dir,
                        scene=scene,
                        method=method,
                        total_images=len(image_paths),
                        ratio=selected_ratio,
                        note=str(exc),
                        project=project,
                    )
                )
                continue

            materialized = None
            subset_source = None
            if materialize:
                materialized = materialize_colmap_subset(
                    scene=scene,
                    image_paths=image_paths,
                    selected_indices=indices,
                    output_dir=work_dir / "fastgs_source",
                    project_root=project,
                    overwrite=overwrite,
                )
                subset_source = materialized.source_path
            else:
                subset_source = scene.root

            model_path = work_dir / "fastgs_output"
            command = fastgs.build_train_command(
                source_path=subset_source.resolve(),
                model_path=model_path.resolve(),
                images=scene.image_dir,
                eval_split=bool(fastgs_options.get("eval_split", True)),
                resolution=fastgs_options.get("resolution"),
                data_device=fastgs_options.get("data_device"),
                extra_args=[str(value) for value in fastgs_options.get("extra_args", [])],
            )
            status = "ready"
            notes = "prepared"
            if materialized and not materialized.runnable:
                status = "needs_colmap_text_model"
                notes = materialized.colmap_status

            prepared.append(
                write_ready_manifest(
                    work_dir=work_dir,
                    scene=scene,
                    method=method,
                    image_names=image_names,
                    indices=indices,
                    ratio=selected_ratio,
                    subset_source=subset_source,
                    command=command,
                    command_text=fastgs.format_command(command),
                    status=status,
                    notes=notes,
                    project=project,
                )
            )

    return {
        "experiment_id": experiment_id,
        "config": relative_to_root(config_source, project),
        "dataset_filter": dataset_id,
        "scene_filters": scene_ids or [],
        "ratio": selected_ratio,
        "methods": selected_methods,
        "output_root": relative_to_root(root, project),
        "prepared": [asdict(run) for run in prepared],
        "warnings": warnings,
    }


def first_ratio(config: dict[str, Any]) -> float:
    ratios = config.get("budgets", {}).get("ratios", [])
    if not ratios:
        raise ValueError("Stage 1 config must define budgets.ratios")
    return float(ratios[0])


def filter_scenes(
    scenes: list[SceneRecord],
    dataset_id: str | None,
    scene_ids: list[str] | None,
) -> list[SceneRecord]:
    scene_filter = set(scene_ids or [])
    return [
        scene
        for scene in scenes
        if (dataset_id is None or scene.dataset_id == dataset_id)
        and (not scene_filter or scene.scene_id in scene_filter)
    ]


def load_method_features(
    scene: SceneRecord,
    method: str,
    image_names: list[str],
) -> list[list[float]] | None:
    if method not in K_CENTER_METHODS:
        return None
    feature_path = method_feature_path(scene, method)
    if feature_path is None:
        raise ValueError(f"{method} requires a feature file in data/datasets.yaml")
    return load_feature_vectors(feature_path, image_names)


def method_feature_path(scene: SceneRecord, method: str) -> Path | None:
    candidates = [method]
    if method in FEATURE_K_CENTER_METHODS:
        candidates.extend(["feature_k_center", "feature", "image_feature"])
    if method in REGISTER_K_CENTER_METHODS:
        candidates.extend(["register_k_center", "register", "vggt_register"])
    for candidate in candidates:
        if candidate in scene.features:
            return scene.features[candidate]
    return None


def write_ready_manifest(
    *,
    work_dir: Path,
    scene: SceneRecord,
    method: str,
    image_names: list[str],
    indices: list[int],
    ratio: float,
    subset_source: Path,
    command: list[str],
    command_text: str,
    status: str,
    notes: str,
    project: Path,
) -> PreparedStage1Run:
    work_dir.mkdir(parents=True, exist_ok=True)
    selected_names = [image_names[index] for index in indices]
    selected_indices_path = work_dir / "selected_indices.txt"
    selected_images_path = work_dir / "selected_images.txt"
    command_path = work_dir / "fastgs_train.sh"
    manifest_path = work_dir / "stage1_subset_manifest.json"

    selected_indices_path.write_text("\n".join(str(index) for index in indices) + "\n", encoding="utf-8")
    selected_images_path.write_text("\n".join(selected_names) + "\n", encoding="utf-8")
    command_path.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{command_text}\n", encoding="utf-8")
    command_path.chmod(0o755)

    payload = {
        "dataset_id": scene.dataset_id,
        "scene_id": scene.scene_id,
        "method": method,
        "status": status,
        "requested_ratio": ratio,
        "actual_ratio": len(indices) / len(image_names),
        "total_images": len(image_names),
        "selected_count": len(indices),
        "selected_indices": relative_to_root(selected_indices_path, project),
        "selected_images": relative_to_root(selected_images_path, project),
        "subset_source": relative_to_root(subset_source, project),
        "fastgs_command_script": relative_to_root(command_path, project),
        "fastgs_command": command,
        "notes": notes,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return PreparedStage1Run(
        method=method,
        dataset_id=scene.dataset_id,
        scene_id=scene.scene_id,
        status=status,
        total_images=len(image_names),
        selected_count=len(indices),
        requested_ratio=ratio,
        actual_ratio=len(indices) / len(image_names),
        work_dir=relative_to_root(work_dir, project),
        selected_indices=relative_to_root(selected_indices_path, project),
        subset_source=relative_to_root(subset_source, project),
        fastgs_command=relative_to_root(command_path, project),
        notes=notes,
    )


def write_pending_manifest(
    *,
    work_dir: Path,
    scene: SceneRecord,
    method: str,
    total_images: int,
    ratio: float,
    note: str,
    project: Path,
) -> PreparedStage1Run:
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "stage1_subset_manifest.json"
    payload = {
        "dataset_id": scene.dataset_id,
        "scene_id": scene.scene_id,
        "method": method,
        "status": "pending",
        "requested_ratio": ratio,
        "total_images": total_images,
        "selected_count": 0,
        "notes": note,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return PreparedStage1Run(
        method=method,
        dataset_id=scene.dataset_id,
        scene_id=scene.scene_id,
        status="pending",
        total_images=total_images,
        selected_count=0,
        requested_ratio=ratio,
        actual_ratio=0.0,
        work_dir=relative_to_root(work_dir, project),
        selected_indices=None,
        subset_source=None,
        fastgs_command=None,
        notes=note,
    )


def ratio_slug(ratio: float) -> str:
    return f"ratio_{int(round(ratio * 100)):03d}"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
