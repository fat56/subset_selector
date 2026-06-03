"""Project path helpers used by lightweight management commands."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing project markers."""

    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "pyproject.toml").exists() and (path / "docs").exists():
            return path
    return current


def relative_to_root(path: Path, root: Path) -> str:
    """Format a path relative to the project root when possible."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

