"""Integration helpers for the sibling VGGT-OMEGA repository.

This module intentionally avoids importing torch or vggt_omega at module import
time. Heavy dependencies are checked or used through the VGGT-OMEGA virtual
environment in a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vggt_omega_selector.project import find_project_root, relative_to_root


CHECKPOINT_ALIASES = {
    "512": "vggt_omega_1b_512.pt",
    "default_512": "vggt_omega_1b_512.pt",
    "vggt_omega_1b_512": "vggt_omega_1b_512.pt",
    "256_text": "vggt_omega_1b_256_text.pt",
    "text_256": "vggt_omega_1b_256_text.pt",
    "vggt_omega_1b_256_text": "vggt_omega_1b_256_text.pt",
}


@dataclass(frozen=True)
class VGGTOmegaIntegration:
    """Resolved paths for the local VGGT-OMEGA integration."""

    root: Path
    python: Path
    project_root: Path

    @classmethod
    def discover(
        cls,
        root: str | Path | None = None,
        python: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "VGGTOmegaIntegration":
        project = Path(project_root).resolve() if project_root else find_project_root()
        resolved_root = resolve_vggt_root(project, root)
        resolved_python = resolve_vggt_python(resolved_root, python, project)
        return cls(root=resolved_root, python=resolved_python, project_root=project)

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def selector_src(self) -> Path:
        return self.project_root / "src"

    def checkpoint_path(self, checkpoint: str | Path | None = None) -> Path:
        if checkpoint is None:
            checkpoint = "512"
        checkpoint_text = str(checkpoint)
        if checkpoint_text in CHECKPOINT_ALIASES:
            return self.checkpoint_dir / CHECKPOINT_ALIASES[checkpoint_text]
        path = Path(checkpoint_text)
        if not path.is_absolute():
            candidate = self.project_root / path
            if candidate.exists():
                return candidate.resolve()
            return (self.checkpoint_dir / path).resolve()
        return path.resolve()

    def subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        additions = [str(self.selector_src), str(self.root)]
        existing = env.get("PYTHONPATH")
        if existing:
            additions.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(additions)
        return env

    def preflight(self) -> dict[str, Any]:
        checkpoints = []
        for alias in ("512", "256_text"):
            path = self.checkpoint_path(alias)
            checkpoints.append(
                {
                    "alias": alias,
                    "path": relative_to_root(path, self.project_root),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                }
            )

        return {
            "root": relative_to_root(self.root, self.project_root),
            "root_resolved": str(self.root),
            "root_exists": self.root.exists(),
            "python": str(self.python),
            "python_exists": self.python.exists(),
            "checkpoint_dir": relative_to_root(self.checkpoint_dir, self.project_root),
            "checkpoints": checkpoints,
            "git_commit": git_output(self.root, ["git", "rev-parse", "--short", "HEAD"]),
            "git_dirty": bool(git_output(self.root, ["git", "status", "--porcelain"])),
            "import_probe": self.import_probe(),
        }

    def import_probe(self) -> dict[str, Any]:
        code = r"""
import json
import torch
import torchvision
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images

print(json.dumps({
    "ok": True,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_available": torch.cuda.is_available(),
    "model_class": f"{VGGTOmega.__module__}.{VGGTOmega.__name__}",
    "load_fn": load_and_preprocess_images.__name__,
}))
"""
        completed = subprocess.run(
            [str(self.python), "-c", code],
            cwd=self.root,
            env=self.subprocess_env(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            return {
                "ok": False,
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
                "stdout": completed.stdout.strip(),
            }
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        try:
            return json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            return {
                "ok": False,
                "returncode": completed.returncode,
                "parse_error": str(exc),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }

    def run_cache(
        self,
        *,
        image_paths: list[str],
        image_list: str | None,
        output_dir: str,
        checkpoint: str,
        image_resolution: int,
        mode: str,
        device: str,
        include_pose: bool,
        include_depth: bool,
        include_images: bool,
        enable_alignment: bool,
        strict_load: bool,
    ) -> int:
        checkpoint_path = self.checkpoint_path(checkpoint)
        command = [
            str(self.python),
            "-m",
            "vggt_omega_selector.tools.vggt_cache_runner",
            "--vggt-root",
            str(self.root),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            output_dir,
            "--image-resolution",
            str(image_resolution),
            "--mode",
            mode,
            "--device",
            device,
        ]
        for image_path in image_paths:
            command.extend(["--image", image_path])
        if image_list:
            command.extend(["--image-list", image_list])
        if include_pose:
            command.append("--include-pose")
        if include_depth:
            command.append("--include-depth")
        if include_images:
            command.append("--include-images")
        if enable_alignment:
            command.append("--enable-alignment")
        if strict_load:
            command.append("--strict-load")

        completed = subprocess.run(command, cwd=self.project_root, env=self.subprocess_env(), check=False)
        return completed.returncode


def resolve_vggt_root(project_root: Path, root: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if root:
        candidates.append(Path(root))
    env_root = os.environ.get("VGGT_OMEGA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            project_root / "external" / "vggt-omega",
            project_root.parent / "vggt-omega",
            Path("/home/m/project/ltm/vggt-omega"),
        ]
    )
    for candidate in candidates:
        resolved = (project_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if (resolved / "vggt_omega").exists():
            return resolved
    return candidates[0].resolve() if candidates else (project_root / "external" / "vggt-omega").resolve()


def resolve_vggt_python(root: Path, python: str | Path | None = None, project_root: Path | None = None) -> Path:
    candidates: list[Path] = []
    if python:
        candidates.append(Path(python))
    env_python = os.environ.get("VGGT_OMEGA_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    candidates.extend([root / ".venv" / "bin" / "python", Path(sys.executable)])
    for candidate in candidates:
        if candidate.is_absolute():
            resolved = candidate
        elif project_root is not None and (project_root / candidate).exists():
            resolved = project_root / candidate
        else:
            resolved = root / candidate
        if resolved.exists():
            return resolved
    fallback = candidates[0]
    return fallback if fallback.is_absolute() else root / fallback


def git_output(cwd: Path, command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    return completed.stdout.strip()
