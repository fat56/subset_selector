"""Small project-management CLI for experiment and run records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vggt_omega_selector.backbones.vggt_omega import VGGTOmegaIntegration
from vggt_omega_selector.project import find_project_root, relative_to_root


METRICS_TEMPLATE = {
    "embedding": {
        "register_cosine_similarity": None,
        "positive_cosine_loss": None,
        "retrieval_top1_accuracy": None,
    },
    "reconstruction": {
        "psnr": None,
        "ssim": None,
        "lpips": None,
    },
    "geometry": {
        "pose_ate": None,
        "pose_rpe": None,
        "depth_abs_rel": None,
    },
    "efficiency": {
        "subset_size": None,
        "subset_ratio": None,
        "vggt_seconds": None,
        "gs_train_seconds": None,
    },
    "correlation": {
        "spearman_rho_register_cosine_vs_psnr": None,
        "pearson_r_register_cosine_vs_psnr": None,
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage VGGT-OMEGA selector experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_exp = subparsers.add_parser("new-experiment", help="Create docs and config for a new experiment.")
    new_exp.add_argument("--id", required=True, help="Experiment id, e.g. 0002_feature_kcenter_ablation.")
    new_exp.add_argument("--title", required=True, help="Human-readable experiment title.")
    new_exp.add_argument("--stage", required=True, help="Stage id, e.g. stage1.")
    new_exp.add_argument("--status", default="planned")
    new_exp.add_argument("--owner", default="")
    new_exp.set_defaults(func=cmd_new_experiment)

    record = subparsers.add_parser("record-run", help="Create a local run manifest and append the run ledger.")
    record.add_argument("--experiment", required=True, help="Experiment id.")
    record.add_argument("--stage", required=True, help="Stage id.")
    record.add_argument("--method", required=True, help="Selection/training method id.")
    record.add_argument("--dataset", required=True, help="Dataset, scene, or scene-set id.")
    record.add_argument("--config", required=True, help="Config path to snapshot.")
    record.add_argument("--run-id", default=None, help="Optional run id. Defaults to timestamp plus method.")
    record.add_argument("--notes", default="", help="Short note stored in notes.md and run ledger.")
    record.add_argument("--status", default="planned")
    record.set_defaults(func=cmd_record_run)

    preflight = subparsers.add_parser("vggt-preflight", help="Check the local VGGT-OMEGA integration.")
    preflight.add_argument("--root", default=None, help="Optional VGGT-OMEGA root override.")
    preflight.add_argument("--python", default=None, help="Optional VGGT-OMEGA Python interpreter override.")
    preflight.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    preflight.add_argument("--strict", action="store_true", help="Exit non-zero if any required check fails.")
    preflight.set_defaults(func=cmd_vggt_preflight)

    cache = subparsers.add_parser("vggt-cache", help="Run VGGT-OMEGA and cache tensor outputs.")
    cache.add_argument("--root", default=None, help="Optional VGGT-OMEGA root override.")
    cache.add_argument("--python", default=None, help="Optional VGGT-OMEGA Python interpreter override.")
    cache.add_argument("--image", action="append", default=[], help="Image path. Can be repeated.")
    cache.add_argument("--images", nargs="*", default=[], help="Image paths.")
    cache.add_argument("--image-list", default=None, help="Text file with one image path per line.")
    cache.add_argument("--output-dir", required=True, help="Output cache directory.")
    cache.add_argument("--checkpoint", default="512", help="Checkpoint alias or path. Aliases: 512, 256_text.")
    cache.add_argument("--image-resolution", type=int, default=512)
    cache.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    cache.add_argument("--device", default="auto")
    cache.add_argument("--include-pose", action="store_true")
    cache.add_argument("--include-depth", action="store_true")
    cache.add_argument("--include-images", action="store_true")
    cache.add_argument("--enable-alignment", action="store_true")
    cache.add_argument("--strict-load", action="store_true")
    cache.set_defaults(func=cmd_vggt_cache)

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_new_experiment(args: argparse.Namespace) -> int:
    root = find_project_root()
    exp_dir = root / "docs" / "experiments" / args.id
    config_path = root / "configs" / "experiments" / f"{args.id}.yaml"
    created = utc_now()

    exp_dir.mkdir(parents=True, exist_ok=False)
    write_text(
        exp_dir / "proposal.md",
        f"""# {args.title}

## Metadata

- Experiment ID: `{args.id}`
- Stage: `{args.stage}`
- Status: {args.status}
- Owner: {args.owner}
- Created: {created[:10]}
- Config: [configs/experiments/{args.id}.yaml](../../../configs/experiments/{args.id}.yaml)

## Question

What exact uncertainty does this experiment reduce?

## Hypothesis

State the expected outcome and the reason it should happen.

## Method

Describe datasets, selection methods, budgets, backbone/readout assumptions, and evaluation backend.

## Metrics

Primary metric:

Secondary metrics:

## Decision Rule

What result changes the next step?

## Risks

Known failure modes and mitigations.
""",
    )
    copy_template(root, "experiment_runbook.md", exp_dir / "runbook.md")
    copy_template(root, "experiment_results.md", exp_dir / "results.md")
    copy_template(root, "experiment_review.md", exp_dir / "review.md")
    write_text(
        config_path,
        f"""experiment:
  id: {args.id}
  title: {args.title}
  stage: {args.stage}
  status: {args.status}
  owner: {args.owner}
  created_at: {created}

question: TBD

datasets:
  registry: data/datasets.yaml
  scenes: TBD

methods: []

budgets:
  fixed_k: []
  ratios: []

evaluation:
  metrics: []

gate:
  decision: pending
""",
    )

    append_experiment_registry(root, args, created, config_path)
    print(f"Created experiment docs: {relative_to_root(exp_dir, root)}")
    print(f"Created config: {relative_to_root(config_path, root)}")
    return 0


def cmd_record_run(args: argparse.Namespace) -> int:
    root = find_project_root()
    config = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    if not config.exists():
        raise SystemExit(f"Config not found: {config}")

    timestamp = utc_now()
    run_id = args.run_id or f"{timestamp.replace(':', '').replace('-', '').replace('T', '_').replace('+0000', 'Z')}_{args.method}"
    run_dir = root / "runs" / args.experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_snapshot = run_dir / "config.yaml"
    shutil.copy2(config, config_snapshot)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(METRICS_TEMPLATE, indent=2) + "\n", encoding="utf-8")

    selected_indices = run_dir / "selected_indices.txt"
    selected_indices.write_text("# one selected frame index per line\n", encoding="utf-8")

    notes_path = run_dir / "notes.md"
    notes_path.write_text(f"# Notes\n\n{args.notes}\n", encoding="utf-8")

    commit = git_output(root, ["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    dirty = bool(git_output(root, ["git", "status", "--porcelain"]))
    config_hash = sha256_file(config_snapshot)
    manifest_path = run_dir / "manifest.yaml"
    write_text(
        manifest_path,
        f"""run:
  id: {run_id}
  experiment_id: {args.experiment}
  timestamp: {timestamp}
  stage: {args.stage}
  method: {args.method}
  dataset: {args.dataset}
  status: {args.status}

reproducibility:
  git_commit: {commit}
  git_dirty: {str(dirty).lower()}
  config: {relative_to_root(config_snapshot, root)}
  config_source: {relative_to_root(config, root)}
  config_sha256: {config_hash}

artifacts:
  selected_indices: {relative_to_root(selected_indices, root)}
  metrics: {relative_to_root(metrics_path, root)}
  notes: {relative_to_root(notes_path, root)}
  large_artifacts: []

decision:
  primary_metric: null
  primary_value: null
  gate: null
""",
    )

    append_run_ledger(root, args, run_id, timestamp, commit, dirty, config_snapshot, manifest_path)
    print(f"Recorded run: {relative_to_root(run_dir, root)}")
    print(f"Manifest: {relative_to_root(manifest_path, root)}")
    return 0


def cmd_vggt_preflight(args: argparse.Namespace) -> int:
    integration = VGGTOmegaIntegration.discover(root=args.root, python=args.python)
    report = integration.preflight()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_preflight_report(report)
    if args.strict and not preflight_ok(report):
        return 1
    return 0


def cmd_vggt_cache(args: argparse.Namespace) -> int:
    image_paths = list(args.image) + list(args.images)
    if not image_paths and not args.image_list:
        raise SystemExit("No images provided. Use --image, --images, or --image-list.")
    integration = VGGTOmegaIntegration.discover(root=args.root, python=args.python)
    return integration.run_cache(
        image_paths=image_paths,
        image_list=args.image_list,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        image_resolution=args.image_resolution,
        mode=args.mode,
        device=args.device,
        include_pose=args.include_pose,
        include_depth=args.include_depth,
        include_images=args.include_images,
        enable_alignment=args.enable_alignment,
        strict_load=args.strict_load,
    )


def copy_template(root: Path, template_name: str, destination: Path) -> None:
    source = root / "docs" / "templates" / template_name
    shutil.copy2(source, destination)


def append_experiment_registry(root: Path, args: argparse.Namespace, created: str, config_path: Path) -> None:
    registry = root / "docs" / "registry" / "experiment_registry.csv"
    fieldnames = [
        "experiment_id",
        "title",
        "stage",
        "status",
        "created_at",
        "primary_doc",
        "config",
        "latest_run",
        "decision",
    ]
    ensure_csv_header(registry, fieldnames)
    with registry.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": args.id,
                "title": args.title,
                "stage": args.stage,
                "status": args.status,
                "created_at": created,
                "primary_doc": f"docs/experiments/{args.id}/proposal.md",
                "config": relative_to_root(config_path, root),
                "latest_run": "",
                "decision": "pending",
            }
        )


def append_run_ledger(
    root: Path,
    args: argparse.Namespace,
    run_id: str,
    timestamp: str,
    commit: str,
    dirty: bool,
    config_snapshot: Path,
    manifest_path: Path,
) -> None:
    ledger = root / "docs" / "registry" / "run_ledger.csv"
    fieldnames = [
        "run_id",
        "experiment_id",
        "timestamp",
        "stage",
        "method",
        "dataset",
        "config",
        "git_commit",
        "git_dirty",
        "status",
        "primary_metric",
        "primary_value",
        "manifest",
        "notes",
    ]
    ensure_csv_header(ledger, fieldnames)
    with ledger.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(
            {
                "run_id": run_id,
                "experiment_id": args.experiment,
                "timestamp": timestamp,
                "stage": args.stage,
                "method": args.method,
                "dataset": args.dataset,
                "config": relative_to_root(config_snapshot, root),
                "git_commit": commit,
                "git_dirty": str(dirty).lower(),
                "status": args.status,
                "primary_metric": "",
                "primary_value": "",
                "manifest": relative_to_root(manifest_path, root),
                "notes": args.notes,
            }
        )


def ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()


def print_preflight_report(report: dict) -> None:
    print("VGGT-OMEGA integration")
    print(f"  root: {report['root']} ({'ok' if report['root_exists'] else 'missing'})")
    print(f"  python: {report['python']} ({'ok' if report['python_exists'] else 'missing'})")
    print(f"  checkpoint_dir: {report['checkpoint_dir']}")
    print(f"  git_commit: {report.get('git_commit') or 'unknown'}")
    print(f"  git_dirty: {str(report.get('git_dirty')).lower()}")
    probe = report.get("import_probe", {})
    if probe.get("ok"):
        print(
            "  import: ok "
            f"(torch {probe.get('torch')}, torchvision {probe.get('torchvision')}, "
            f"cuda={str(probe.get('cuda_available')).lower()})"
        )
    else:
        print("  import: failed")
        if probe.get("stderr"):
            print(f"    stderr: {probe['stderr']}")
    print("  checkpoints:")
    for checkpoint in report.get("checkpoints", []):
        size = checkpoint.get("size_bytes")
        size_gb = f"{size / (1024 ** 3):.2f} GiB" if size else "missing"
        print(f"    - {checkpoint['alias']}: {checkpoint['path']} ({size_gb})")


def preflight_ok(report: dict) -> bool:
    return (
        bool(report.get("root_exists"))
        and bool(report.get("python_exists"))
        and bool(report.get("import_probe", {}).get("ok"))
        and all(item.get("exists") for item in report.get("checkpoints", []))
    )


def git_output(root: Path, command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
