# Runbook

## Preconditions

- `data/datasets.yaml` 已登记真实数据。
- 每个 scene 有 FastGS/3DGS 可读的 `images/` 和 COLMAP `sparse/0` text 或 binary model。
- VGGT-OMEGA checkpoint 和 FastGS backend 路径已在运行环境中可用。
- `configs/experiments/0001_stage1_register_quality_gate.yaml` 已冻结本次 20% ratio 和方法列表。

## Planned Flow

0. Run VGGT-OMEGA integration preflight:

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage vggt-preflight --strict
```

1. Run FastGS integration preflight:

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage fastgs-preflight --strict
```

2. 为每个 scene 生成 20% baseline 子集和 FastGS source/command：

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage stage1-prepare \
  --config configs/experiments/0001_stage1_register_quality_gate.yaml \
  --dataset 3dgsdata \
  --overwrite
```

3. 缓存 full-set 和 subset 的 register/readout embedding。
4. 执行每个 prepared run 下的 `fastgs_train.sh`。
5. 写入 `runs/.../metrics.json`。
6. 汇总散点和 Spearman/Pearson 到 `results.md`。

## Record A Run

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage record-run \
  --experiment 0001_stage1_register_quality_gate \
  --stage stage1 \
  --method random_ratio \
  --dataset <dataset_or_scene_set> \
  --config configs/experiments/0001_stage1_register_quality_gate.yaml \
  --notes "20 percent FastGS baseline dry run"
```

## Required Artifacts

- `selected_indices.txt` or `selected_indices.json`
- `stage1_subset_manifest.json`
- `fastgs_source/`
- `fastgs_train.sh`
- VGGT cache `manifest.json`
- `camera_and_register_tokens.pt` / `register_tokens.pt`
- `metrics.json`
- `manifest.yaml`
- Optional scatter plot path recorded in manifest
