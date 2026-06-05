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

1.3. 对 random/uniform prepared runs，split 必须先在 full image set 上按 llffhold 计算 test set，再从非 test pool 中选择 train subset。`scripts/run_fastgs_random_uniform_queue.sh prepare` 会校验 `stage1_split.json`、full-scene test set、selected train set 和 train/test disjoint。

2. 为每个 scene 生成 20% baseline 子集和 FastGS source/command：

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage stage1-prepare \
  --config configs/experiments/0001_stage1_register_quality_gate.yaml \
  --dataset 3dgsdata \
  --overwrite
```

3. 缓存 full-set 和 subset 的 register embedding，并计算 scene 内相似度/质量相关性：

```bash
PYTHONPATH=src python scripts/run_stage1_register_similarity.py
```

该脚本会对每个 scene 缓存 full-train(non-test) reference、5 个 random subset 和 1 个 uniform subset 的 VGGT-OMEGA register tokens，写入 `caches/vggt_omega/0001_stage1_register_quality_gate/register_similarity_images512`，并输出文档 CSV 到 `docs/experiments/0001_stage1_register_quality_gate/register_similarity/`。

4. 执行 FastGS 训练。random/uniform 的 `images_4` 30k 矩阵可用队列脚本：

```bash
bash scripts/run_fastgs_random_uniform_queue.sh prepare
bash scripts/run_fastgs_random_uniform_queue.sh launch
bash scripts/run_fastgs_random_uniform_queue.sh status
```

该脚本默认写入每个 run 下的 `fastgs_output_images4_30k`，使用 `--images images_4 --iterations 30000 --densification_interval 100`。

5. 写入 FastGS `results.json`，必要时同步到项目级 `metrics.json`。
6. 汇总 scene 内 Spearman/Pearson 到 `results.md`。

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
- `register_mean_embedding.json`
- `register_similarity/subset_register_similarity.csv`
- `register_similarity/scene_register_correlations.csv`
- FastGS `results.json` 或项目级 `metrics.json`
- `manifest.yaml`
- 可选：在 manifest 中记录散点图路径。
