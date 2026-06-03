# Runbook

## Preconditions

- Dataset registry is updated in `data/datasets.yaml`.
- Config is frozen in `configs/experiments/<id>.yaml`.
- Expected outputs and metrics are known.

## Commands

```bash
# Replace placeholders before running.
PYTHONPATH=src python -m vggt_omega_selector.cli.manage record-run \
  --experiment <id> \
  --stage <stage> \
  --method <method> \
  --dataset <dataset_or_scene_set> \
  --config configs/experiments/<id>.yaml \
  --notes "what changed"
```

## Artifacts

Record large artifact paths in `runs/<id>/<run_id>/manifest.yaml`; do not copy large files into docs.

## Checks

- Metrics file populated.
- Selected indices saved.
- Run ledger row appended.
- Results summary updated.
