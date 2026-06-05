# 运行手册

## 前置条件

- `data/datasets.yaml` 已登记真实数据。
- 每个 scene 有 FastGS/3DGS 可读的 `images/` 和 COLMAP `sparse/0` text 或 binary model。
- VGGT-OMEGA checkpoint 和 FastGS backend 路径已在运行环境中可用。
- `configs/experiments/0001_stage1_register_quality_gate.yaml` 已冻结本次 20% ratio 和方法列表。

## 计划流程

0. 运行 VGGT-OMEGA 集成预检：

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage vggt-preflight --strict
```

1. 运行 FastGS 集成预检：

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage fastgs-preflight --strict
```

1.1. 在 RTX 5090 / CUDA 12.8 环境中，确认 FastGS rasterizer fix1 已应用并重编译。至少先跑一条 bonsai 30k sanity check；未应用 fix1 时已观测到 `diff_gaussian_rasterization_fastgs._C.rasterize_gaussians` forward 在长训中触发 CUDA illegal memory access。

1.2. 对 Stage 1 prepared source，确认 FastGS COLMAP reader 会优先使用 `stage1_split.json`。日志应出现类似 `Using stage1_split.json: 51 train cameras, 37 test cameras`；不能让原生 `--eval` 对物化后的 source 重新 llffhold。

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

## 记录一次运行

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage record-run \
  --experiment 0001_stage1_register_quality_gate \
  --stage stage1 \
  --method random_ratio \
  --dataset <dataset_or_scene_set> \
  --config configs/experiments/0001_stage1_register_quality_gate.yaml \
  --notes "20 percent FastGS baseline dry run"
```

## 必需产物

- `selected_indices.txt` 或 `selected_indices.json`
- `stage1_subset_manifest.json`
- `fastgs_source/`
- `fastgs_train.sh`
- VGGT cache `manifest.json`
- `camera_and_register_tokens.pt` / `register_tokens.pt`
- `metrics.json`
- `manifest.yaml`
- 可选：在 manifest 中记录散点图路径。
