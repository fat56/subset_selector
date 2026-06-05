"""Ratio-based baseline selectors for Stage 1 experiments."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence


RANDOM_METHODS = {"random", "random_k", "random_ratio"}
UNIFORM_METHODS = {"uniform", "uniform_stride", "uniform_stride_k", "uniform_stride_ratio"}
FEATURE_K_CENTER_METHODS = {"feature_k_center", "feature_k_center_ratio"}
REGISTER_K_CENTER_METHODS = {"register_k_center", "register_k_center_ratio"}
K_CENTER_METHODS = FEATURE_K_CENTER_METHODS | REGISTER_K_CENTER_METHODS


def subset_size_for_ratio(total: int, ratio: float, *, min_count: int = 1) -> int:
    """Return the Stage 1 subset size for a dataset ratio."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        return 0
    if ratio <= 0 or ratio > 1:
        raise ValueError("ratio must be in the interval (0, 1]")
    return min(total, max(min_count, math.ceil(total * ratio)))


def random_ratio_indices(total: int, ratio: float, *, seed: int = 0) -> list[int]:
    """Select a deterministic random subset and return indices in dataset order."""

    count = subset_size_for_ratio(total, ratio)
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), count))


def uniform_stride_ratio_indices(total: int, ratio: float) -> list[int]:
    """Select evenly spaced indices for a ratio budget."""

    count = subset_size_for_ratio(total, ratio)
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]

    step = total / count
    indices = [min(total - 1, int((index + 0.5) * step)) for index in range(count)]
    return sorted(dict.fromkeys(indices))


def k_center_ratio_indices(vectors: Sequence[Sequence[float]], ratio: float) -> list[int]:
    """Farthest-first k-center over precomputed per-image vectors."""

    total = len(vectors)
    count = subset_size_for_ratio(total, ratio)
    if count >= total:
        return list(range(total))
    if count == 0:
        return []

    dense_vectors = [tuple(float(value) for value in vector) for vector in vectors]
    dimensions = {len(vector) for vector in dense_vectors}
    if len(dimensions) != 1:
        raise ValueError("all feature vectors must have the same dimensionality")

    centroid = [
        sum(vector[dimension] for vector in dense_vectors) / total
        for dimension in range(next(iter(dimensions)))
    ]
    first = min(range(total), key=lambda index: squared_distance(dense_vectors[index], centroid))
    selected = [first]
    min_distances = [squared_distance(vector, dense_vectors[first]) for vector in dense_vectors]

    while len(selected) < count:
        candidate = max(
            (index for index in range(total) if index not in selected),
            key=lambda index: (min_distances[index], -index),
        )
        selected.append(candidate)
        for index, vector in enumerate(dense_vectors):
            min_distances[index] = min(min_distances[index], squared_distance(vector, dense_vectors[candidate]))

    return sorted(selected)


def select_ratio_indices(
    method: str,
    total: int,
    ratio: float,
    *,
    seed: int = 0,
    feature_vectors: Sequence[Sequence[float]] | None = None,
) -> list[int]:
    """Dispatch a Stage 1 method id to a ratio-based selector."""

    if method in RANDOM_METHODS:
        return random_ratio_indices(total, ratio, seed=seed)
    if method in UNIFORM_METHODS:
        return uniform_stride_ratio_indices(total, ratio)
    if method in K_CENTER_METHODS:
        if feature_vectors is None:
            raise ValueError(f"{method} requires precomputed feature vectors")
        if len(feature_vectors) != total:
            raise ValueError(f"{method} expected {total} feature vectors, found {len(feature_vectors)}")
        return k_center_ratio_indices(feature_vectors, ratio)
    raise ValueError(f"unsupported Stage 1 selection method: {method}")


def load_feature_vectors(path: str | Path, image_names: Sequence[str]) -> list[list[float]]:
    """Load feature vectors from a small JSON file.

    Supported layouts:
    - ``[[...], [...]]`` aligned to the scene image list.
    - ``{"image_names": [...], "vectors": [[...], ...]}``.
    - ``{"relative/or/base/name.jpg": [...]}``.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [coerce_vector(vector) for vector in payload]
    if not isinstance(payload, dict):
        raise ValueError(f"feature file must be a JSON list or object: {source}")

    if "vectors" in payload:
        vectors = [coerce_vector(vector) for vector in payload["vectors"]]
        names = payload.get("image_names")
        if names is None:
            return vectors
        by_name = {str(name): vector for name, vector in zip(names, vectors)}
        return vectors_for_image_names(by_name, image_names, source)

    by_name = {str(name): coerce_vector(vector) for name, vector in payload.items()}
    return vectors_for_image_names(by_name, image_names, source)


def vectors_for_image_names(
    by_name: dict[str, list[float]],
    image_names: Sequence[str],
    source: Path,
) -> list[list[float]]:
    vectors = []
    for image_name in image_names:
        key = image_name if image_name in by_name else Path(image_name).name
        if key not in by_name:
            raise ValueError(f"missing feature vector for {image_name} in {source}")
        vectors.append(by_name[key])
    return vectors


def coerce_vector(value: object) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("feature vector must be a list")
    return [float(item) for item in value]


def squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right))
