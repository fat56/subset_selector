from __future__ import annotations

import sys
from pathlib import Path

from vggt_omega_selector.baselines.ratio import (
    k_center_ratio_indices,
    random_ratio_indices,
    subset_size_for_ratio,
    uniform_stride_ratio_indices,
)
from vggt_omega_selector.cli.manage import main
from vggt_omega_selector.data.colmap_subset import materialize_colmap_subset
from vggt_omega_selector.data.registry import SceneRecord
from vggt_omega_selector.evaluation.fastgs import FastGSIntegration


def test_ratio_budget_ceil_at_least_one() -> None:
    assert subset_size_for_ratio(10, 0.20) == 2
    assert subset_size_for_ratio(11, 0.20) == 3
    assert subset_size_for_ratio(3, 0.20) == 1
    assert subset_size_for_ratio(0, 0.20) == 0


def test_ratio_baselines_are_deterministic() -> None:
    assert random_ratio_indices(10, 0.20, seed=13) == random_ratio_indices(10, 0.20, seed=13)
    assert uniform_stride_ratio_indices(10, 0.20) == [2, 7]
    assert k_center_ratio_indices([[0.0], [1.0], [10.0], [11.0], [12.0]], 0.40) == [0, 2]


def test_fastgs_command_shape(tmp_path: Path) -> None:
    fastgs_root = tmp_path / "FastGS"
    fastgs_root.mkdir()
    train_script = fastgs_root / "train.py"
    train_script.write_text("# stub\n", encoding="utf-8")

    integration = FastGSIntegration.discover(root=fastgs_root, python=sys.executable)
    command = integration.build_train_command(
        source_path=tmp_path / "source",
        model_path=tmp_path / "model",
        images="images",
        eval_split=True,
    )
    assert str(train_script) in command
    assert "--source_path" in command
    assert "--model_path" in command
    assert "--eval" in command


def test_colmap_text_subset_materializer(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    images = scene_root / "images"
    sparse = scene_root / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    for name in ("000.png", "001.png", "002.png"):
        (images / name).write_text("image\n", encoding="utf-8")
    (sparse / "cameras.txt").write_text(
        "# cameras\n"
        "1 PINHOLE 100 100 50 50 50 50\n"
        "2 PINHOLE 100 100 50 50 50 50\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "# images\n"
        "1 1 0 0 0 0 0 0 1 000.png\n"
        "0 0 7\n"
        "2 1 0 0 0 0 0 0 2 001.png\n"
        "0 0 7\n"
        "3 1 0 0 0 0 0 0 2 002.png\n"
        "0 0 8\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text(
        "# points\n"
        "7 0 0 0 255 255 255 1.0 1 0 2 0\n"
        "8 0 0 0 255 255 255 1.0 3 0\n",
        encoding="utf-8",
    )

    scene = SceneRecord(dataset_id="d", scene_id="s", root=scene_root)
    materialized = materialize_colmap_subset(
        scene=scene,
        image_paths=sorted(images.iterdir()),
        selected_indices=[0, 1],
        output_dir=tmp_path / "prepared",
        project_root=tmp_path,
    )

    assert materialized.runnable
    assert (materialized.images_path / "000.png").is_symlink()
    assert "001.png" in (materialized.sparse_path / "images.txt").read_text(encoding="utf-8")
    assert "002.png" not in (materialized.sparse_path / "images.txt").read_text(encoding="utf-8")


def test_new_cli_help_smoke() -> None:
    for command in ("fastgs-preflight", "stage1-prepare"):
        try:
            main([command, "--help"])
        except SystemExit as exc:
            assert exc.code == 0
