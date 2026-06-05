"""FastGS reconstruction backend integration."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vggt_omega_selector.project import find_project_root, relative_to_root


@dataclass(frozen=True)
class FastGSIntegration:
    """Resolved paths and command construction for FastGS."""

    root: Path
    python: Path
    project_root: Path

    @classmethod
    def discover(
        cls,
        root: str | Path | None = None,
        python: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "FastGSIntegration":
        project = Path(project_root).resolve() if project_root else find_project_root()
        resolved_root = resolve_fastgs_root(project, root)
        resolved_python = resolve_fastgs_python(resolved_root, python, project)
        return cls(root=resolved_root, python=resolved_python, project_root=project)

    @property
    def train_script(self) -> Path:
        return self.root / "train.py"

    def build_train_command(
        self,
        *,
        source_path: str | Path,
        model_path: str | Path,
        images: str = "images",
        eval_split: bool = True,
        resolution: int | None = None,
        data_device: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        command = [
            str(self.python),
            str(self.train_script),
            "--source_path",
            str(source_path),
            "--model_path",
            str(model_path),
            "--images",
            images,
        ]
        if eval_split:
            command.append("--eval")
        if resolution is not None:
            command.extend(["--resolution", str(resolution)])
        if data_device:
            command.extend(["--data_device", data_device])
        if extra_args:
            command.extend(extra_args)
        return command

    def format_command(self, command: list[str]) -> str:
        return shlex.join(command)

    def preflight(self) -> dict[str, Any]:
        return {
            "root": relative_to_root(self.root, self.project_root),
            "root_resolved": str(self.root),
            "root_exists": self.root.exists(),
            "python": str(self.python),
            "python_exists": self.python.exists(),
            "train_script": relative_to_root(self.train_script, self.project_root),
            "train_script_exists": self.train_script.exists(),
            "git_commit": git_output(self.root, ["git", "rev-parse", "--short", "HEAD"]),
            "git_dirty": bool(git_output(self.root, ["git", "status", "--porcelain"])),
            "import_probe": self.import_probe(),
        }

    def import_probe(self) -> dict[str, Any]:
        code = r"""
import json
import numpy
import torch
import torchvision
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

devices = []
for index in range(torch.cuda.device_count()):
    devices.append({
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "capability": list(torch.cuda.get_device_capability(index)),
    })

print(json.dumps({
    "ok": True,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": devices,
    "extensions": {
        "simple_knn": distCUDA2.__name__,
        "fused_ssim": fused_ssim.__name__,
        "diff_gaussian_rasterization_fastgs": GaussianRasterizer.__name__,
    },
}))
"""
        completed = subprocess.run(
            [str(self.python), "-c", code],
            cwd=self.root,
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
            import json

            return json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            return {
                "ok": False,
                "returncode": completed.returncode,
                "parse_error": str(exc),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }


def resolve_fastgs_root(project_root: Path, root: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if root:
        candidates.append(Path(root))
    env_root = os.environ.get("FASTGS_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            project_root / "external" / "FastGS",
            project_root / "external" / "fastgs",
            project_root.parent / "FastGS",
            project_root.parent / "fastgs",
        ]
    )
    for candidate in candidates:
        resolved = (project_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if (resolved / "train.py").exists():
            return resolved
    return candidates[0].resolve() if candidates else (project_root / "external" / "FastGS").resolve()


def resolve_fastgs_python(root: Path, python: str | Path | None = None, project_root: Path | None = None) -> Path:
    candidates: list[Path] = []
    if python:
        candidates.append(Path(python))
    env_python = os.environ.get("FASTGS_PYTHON")
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


def fastgs_preflight_ok(report: dict[str, Any]) -> bool:
    return (
        bool(report.get("root_exists"))
        and bool(report.get("python_exists"))
        and bool(report.get("train_script_exists"))
        and bool(report.get("import_probe", {}).get("ok"))
    )


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
