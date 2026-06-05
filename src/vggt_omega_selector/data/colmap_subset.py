"""Materialize selected-image COLMAP scene subsets for GS backends."""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vggt_omega_selector.data.registry import SceneRecord
from vggt_omega_selector.project import relative_to_root


@dataclass(frozen=True)
class MaterializedSubset:
    """Paths and status for a prepared sparse-view source directory."""

    source_path: Path
    images_path: Path
    sparse_path: Path
    colmap_status: str
    runnable: bool


def materialize_colmap_subset(
    *,
    scene: SceneRecord,
    image_paths: list[Path],
    selected_indices: list[int],
    output_dir: Path,
    project_root: Path,
    overwrite: bool = False,
) -> MaterializedSubset:
    """Create a FastGS-compatible source directory for selected scene images."""

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"subset source already exists: {output_dir}")
        shutil.rmtree(output_dir)

    selected_paths = [image_paths[index] for index in selected_indices]
    images_out = output_dir / scene.image_dir
    sparse_out = output_dir / scene.sparse_dir
    images_out.mkdir(parents=True, exist_ok=True)

    selected_rel_paths = []
    for image_path in selected_paths:
        rel_path = image_path.relative_to(scene.images_path)
        selected_rel_paths.append(rel_path.as_posix())
        destination = images_out / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(image_path.resolve(), destination)

    (output_dir / "selected_images.json").write_text(
        json.dumps(
            {
                "scene": scene.scene_id,
                "image_dir": scene.image_dir,
                "selected_indices": selected_indices,
                "selected_images": selected_rel_paths,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    colmap_status = materialize_sparse_text_model(
        source_sparse=scene.sparse_path,
        destination_sparse=sparse_out,
        selected_image_names=selected_rel_paths,
    )
    runnable = colmap_status in {"filtered_text_model", "filtered_binary_model"}
    if not runnable:
        write_subset_note(output_dir, scene, colmap_status, project_root)

    return MaterializedSubset(
        source_path=output_dir,
        images_path=images_out,
        sparse_path=sparse_out,
        colmap_status=colmap_status,
        runnable=runnable,
    )


def materialize_sparse_text_model(
    *,
    source_sparse: Path,
    destination_sparse: Path,
    selected_image_names: list[str],
) -> str:
    cameras = source_sparse / "cameras.txt"
    images = source_sparse / "images.txt"
    points = source_sparse / "points3D.txt"
    cameras_bin = source_sparse / "cameras.bin"
    images_bin = source_sparse / "images.bin"
    points_bin = source_sparse / "points3D.bin"
    if not source_sparse.exists():
        return "missing_sparse_model"
    destination_sparse.mkdir(parents=True, exist_ok=True)
    if cameras.exists() and images.exists() and points.exists():
        image_records, selected_image_ids, selected_camera_ids = filter_image_records(images, selected_image_names)
        retained_point_ids = filter_points3d(points, destination_sparse / "points3D.txt", selected_image_ids)
        write_filtered_cameras(cameras, destination_sparse / "cameras.txt", selected_camera_ids)
        write_filtered_images(destination_sparse / "images.txt", image_records, retained_point_ids)
        return "filtered_text_model"
    if cameras_bin.exists() and images_bin.exists() and points_bin.exists():
        materialize_sparse_binary_model(
            cameras_bin=cameras_bin,
            images_bin=images_bin,
            points_bin=points_bin,
            destination_sparse=destination_sparse,
            selected_image_names=selected_image_names,
        )
        return "filtered_binary_model"
    return "requires_colmap_sparse_model"


def filter_image_records(
    source: Path,
    selected_image_names: Iterable[str],
) -> tuple[list[tuple[str, str]], set[str], set[str]]:
    selected = set(selected_image_names)
    selected_basenames = {Path(name).name for name in selected}
    lines = source.read_text(encoding="utf-8").splitlines()
    records: list[tuple[str, str]] = []
    selected_image_ids: set[str] = set()
    selected_camera_ids: set[str] = set()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.startswith("#"):
            index += 1
            continue
        points_line = lines[index + 1] if index + 1 < len(lines) else ""
        parts = line.split(maxsplit=9)
        if len(parts) < 10:
            index += 2
            continue
        image_id = parts[0]
        camera_id = parts[8]
        image_name = parts[9]
        if image_name in selected or ("/" not in image_name and Path(image_name).name in selected_basenames):
            records.append((line, points_line))
            selected_image_ids.add(image_id)
            selected_camera_ids.add(camera_id)
        index += 2
    return records, selected_image_ids, selected_camera_ids


CAMERA_MODEL_NAMES = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def materialize_sparse_binary_model(
    *,
    cameras_bin: Path,
    images_bin: Path,
    points_bin: Path,
    destination_sparse: Path,
    selected_image_names: list[str],
) -> None:
    cameras = read_cameras_binary(cameras_bin)
    image_records, selected_image_ids, selected_camera_ids = read_selected_images_binary(
        images_bin,
        selected_image_names,
    )
    retained_point_ids = filter_points3d_binary(points_bin, destination_sparse / "points3D.txt", selected_image_ids)
    write_binary_cameras_text(destination_sparse / "cameras.txt", cameras, selected_camera_ids)
    write_binary_images_text(destination_sparse / "images.txt", image_records, retained_point_ids)


def read_cameras_binary(path: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    cameras = {}
    with path.open("rb") as handle:
        num_cameras = read_binary(handle, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_binary(handle, "iiQQ")
            if model_id not in CAMERA_MODEL_NAMES:
                raise ValueError(f"unsupported COLMAP camera model id {model_id} in {path}")
            model_name, param_count = CAMERA_MODEL_NAMES[model_id]
            params = list(read_binary(handle, "d" * param_count))
            cameras[int(camera_id)] = (model_name, int(width), int(height), params)
    return cameras


def read_selected_images_binary(
    path: Path,
    selected_image_names: Iterable[str],
) -> tuple[list[tuple[int, list[float], list[float], int, str, list[tuple[float, float, int]]]], set[int], set[int]]:
    selected = set(selected_image_names)
    selected_basenames = {Path(name).name for name in selected}
    records = []
    selected_image_ids: set[int] = set()
    selected_camera_ids: set[int] = set()
    with path.open("rb") as handle:
        num_images = read_binary(handle, "Q")[0]
        for _ in range(num_images):
            unpacked = read_binary(handle, "idddddddi")
            image_id = int(unpacked[0])
            qvec = [float(value) for value in unpacked[1:5]]
            tvec = [float(value) for value in unpacked[5:8]]
            camera_id = int(unpacked[8])
            image_name = read_null_terminated_string(handle)
            point_count = read_binary(handle, "Q")[0]
            points2d = []
            for _point_index in range(point_count):
                x_value, y_value, point3d_id = read_binary(handle, "ddq")
                points2d.append((float(x_value), float(y_value), int(point3d_id)))
            if image_name in selected or ("/" not in image_name and Path(image_name).name in selected_basenames):
                records.append((image_id, qvec, tvec, camera_id, image_name, points2d))
                selected_image_ids.add(image_id)
                selected_camera_ids.add(camera_id)
    return records, selected_image_ids, selected_camera_ids


def filter_points3d_binary(source: Path, destination: Path, selected_image_ids: set[int]) -> set[int]:
    retained_point_ids: set[int] = set()
    output_lines = [
        "# Filtered from COLMAP binary by vggt_omega_selector.",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    with source.open("rb") as handle:
        num_points = read_binary(handle, "Q")[0]
        for _ in range(num_points):
            unpacked = read_binary(handle, "QdddBBBd")
            point_id = int(unpacked[0])
            xyz = [float(value) for value in unpacked[1:4]]
            rgb = [int(value) for value in unpacked[4:7]]
            error = float(unpacked[7])
            track_length = read_binary(handle, "Q")[0]
            retained_track: list[int] = []
            for _track_index in range(track_length):
                image_id, point2d_idx = read_binary(handle, "ii")
                if int(image_id) in selected_image_ids:
                    retained_track.extend([int(image_id), int(point2d_idx)])
            if retained_track:
                retained_point_ids.add(point_id)
                values = [point_id, *xyz, *rgb, error, *retained_track]
                output_lines.append(" ".join(format_colmap_value(value) for value in values))
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return retained_point_ids


def write_binary_cameras_text(
    destination: Path,
    cameras: dict[int, tuple[str, int, int, list[float]]],
    selected_camera_ids: set[int],
) -> None:
    output_lines = [
        "# Filtered from COLMAP binary by vggt_omega_selector.",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for camera_id in sorted(selected_camera_ids):
        model_name, width, height, params = cameras[camera_id]
        values = [camera_id, model_name, width, height, *params]
        output_lines.append(" ".join(format_colmap_value(value) for value in values))
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def write_binary_images_text(
    destination: Path,
    records: list[tuple[int, list[float], list[float], int, str, list[tuple[float, float, int]]]],
    retained_point_ids: set[int],
) -> None:
    output_lines = [
        "# Filtered from COLMAP binary by vggt_omega_selector.",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for image_id, qvec, tvec, camera_id, image_name, points2d in records:
        image_values = [image_id, *qvec, *tvec, camera_id, image_name]
        output_lines.append(" ".join(format_colmap_value(value) for value in image_values))
        point_values = []
        for x_value, y_value, point_id in points2d:
            if point_id != -1 and point_id not in retained_point_ids:
                point_id = -1
            point_values.extend([x_value, y_value, point_id])
        output_lines.append(" ".join(format_colmap_value(value) for value in point_values))
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def filter_points3d(source: Path, destination: Path, selected_image_ids: set[str]) -> set[str]:
    retained_point_ids: set[str] = set()
    output_lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            output_lines.append(line)
            continue
        parts = line.split()
        if len(parts) <= 8:
            continue
        track = parts[8:]
        retained_track = []
        for image_id, point2d_idx in zip(track[0::2], track[1::2]):
            if image_id in selected_image_ids:
                retained_track.extend([image_id, point2d_idx])
        if retained_track:
            retained_point_ids.add(parts[0])
            output_lines.append(" ".join([*parts[:8], *retained_track]))
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return retained_point_ids


def write_filtered_cameras(source: Path, destination: Path, selected_camera_ids: set[str]) -> None:
    output_lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            output_lines.append(line)
            continue
        camera_id = line.split(maxsplit=1)[0]
        if camera_id in selected_camera_ids:
            output_lines.append(line)
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def write_filtered_images(
    destination: Path,
    records: list[tuple[str, str]],
    retained_point_ids: set[str],
) -> None:
    output_lines = [
        "# Filtered by vggt_omega_selector Stage 1 subset materializer.",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for image_line, points_line in records:
        output_lines.append(image_line)
        output_lines.append(filter_points2d_line(points_line, retained_point_ids))
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def filter_points2d_line(line: str, retained_point_ids: set[str]) -> str:
    parts = line.split()
    filtered = []
    for x_value, y_value, point_id in zip(parts[0::3], parts[1::3], parts[2::3]):
        if point_id != "-1" and point_id not in retained_point_ids:
            point_id = "-1"
        filtered.extend([x_value, y_value, point_id])
    return " ".join(filtered)


def read_binary(handle, format_char_sequence: str):
    size = struct.calcsize("<" + format_char_sequence)
    return struct.unpack("<" + format_char_sequence, handle.read(size))


def read_null_terminated_string(handle) -> str:
    chars = []
    while True:
        char = handle.read(1)
        if char == b"\x00" or char == b"":
            break
        chars.append(char)
    return b"".join(chars).decode("utf-8")


def format_colmap_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def write_subset_note(output_dir: Path, scene: SceneRecord, colmap_status: str, project_root: Path) -> None:
    note = f"""# Subset Source Not Runnable Yet

Images were selected and symlinked, but FastGS expects a COLMAP text sparse model at:

```text
{relative_to_root(scene.sparse_path, project_root)}
```

Current sparse status: `{colmap_status}`.

Convert the scene sparse model to text under `sparse/0` before rerunning `stage1-prepare`.
"""
    (output_dir / "SUBSET_NOTE.md").write_text(note, encoding="utf-8")
