# FastGS Integration

Stage 1 uses FastGS as the first GS reconstruction backend.

## Source

- Repository: <https://github.com/fastgs/FastGS>
- Verified HEAD during scaffold setup: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`
- Selector-side config: [configs/integrations/fastgs.yaml](../../configs/integrations/fastgs.yaml)

FastGS is expected as an external checkout, not vendored into this repository:

```text
external/FastGS
```

Override paths with `FASTGS_ROOT`, `FASTGS_PYTHON`, or CLI flags.

## Preflight

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage fastgs-preflight --strict
```

The preflight checks the root, Python interpreter, `train.py`, and git status.

## Command Shape

`stage1-prepare` writes one `fastgs_train.sh` per prepared scene/method. The generated command follows FastGS' `train.py` interface:

```bash
python train.py \
  --source_path <prepared_fastgs_source> \
  --model_path <prepared_fastgs_output> \
  --images images \
  --eval
```

The prepared source is a sparse-view COLMAP scene: selected images are symlinked, and `sparse/0` text or binary sparse models are filtered to a selected-image text model.
