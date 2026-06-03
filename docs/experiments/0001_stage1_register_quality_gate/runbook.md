# Runbook

## Preconditions

- `data/datasets.yaml` 已登记真实数据。
- VGGT-OMEGA checkpoint 和 3DGS/InstantSplat backend 路径已在运行环境中可用。
- `configs/experiments/0001_stage1_register_quality_gate.yaml` 已冻结本次预算和方法列表。

## Planned Flow

0. Run VGGT-OMEGA integration preflight:

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage vggt-preflight --strict
```

1. 为每个 scene 生成 baseline 子集。
2. 缓存 full-set 和 subset 的 register/readout embedding。
3. 对每个子集运行 sparse-view 3DGS/InstantSplat 评测。
4. 写入 `runs/.../metrics.json`。
5. 汇总散点和 Spearman/Pearson 到 `results.md`。

## Record A Run

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage record-run \
  --experiment 0001_stage1_register_quality_gate \
  --stage stage1 \
  --method random_k \
  --dataset <dataset_or_scene_set> \
  --config configs/experiments/0001_stage1_register_quality_gate.yaml \
  --notes "baseline dry run"
```

## Required Artifacts

- `selected_indices.txt` or `selected_indices.json`
- VGGT cache `manifest.json`
- `camera_and_register_tokens.pt` / `register_tokens.pt`
- `metrics.json`
- `manifest.yaml`
- Optional scatter plot path recorded in manifest
