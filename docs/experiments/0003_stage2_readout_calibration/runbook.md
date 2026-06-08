# 运行手册

## 前置条件

- `0001_stage1_register_quality_gate` results 已可用。
- `0002_ltm30_pose_depth_validation` results 已可用，并保持为 held-out validation，除非后续 split 文档明确更新。
- VGGT-OMEGA checkpoint 和 cache runner 已可用。
- `data/raw/ltm_datasets` 指向 `/home/m/dataset/ltm_datasets`。
- Train/val/test split 按 scene 划分；同一 scene 的 subsets 不能跨 split。

## 计划流程

0. 冻结 target definition。

   - Primary native metrics（主 native 指标）: `pose_rotation_mean_deg`, `pointmap_rmse_norm`, `depth_log_rmse`。
   - Secondary sanity metric（辅助 sanity 指标）: sensor `gt_depth_absrel_mean`。
   - Direct sensor `gt_pose_*` 排除在 gate 之外。

1. 构建额外 readout-calibration scenes。

   - 复用 LTM preparation logic。
   - 优先使用同时有 pose 和 depth 的 scenes。
   - 每个 scene 的 full images 上限为 200。
   - 生成 `10%`、`20%`、`30%` 的 candidate subsets。
   - 包含 random、uniform、contiguous、k-center/farthest，以及刻意冗余的 negatives。
   - MVP source pool: WildRGBD + DL3DV，排除当前 LTM30 validation scenes。
   - 在 depth+pose readout baseline 验证前，ScanNet pose-only scenes 保持 optional。

2. 冻结 scene splits。

   - `readout_train`: 300-500 scenes，与 LTM30 无重叠。
   - `readout_val`: 当前 LTM30，加可选额外 held-out scenes。
   - `selector_train`: 为 `0004` 保留，最好不与 `readout_train` 有 scene 重叠。
   - `selector_val/test`: 为 hard VGGT/FastGS/VLA validation 保留。
   - 在 VGGT cache generation 前记录 split membership。

3. 缓存 VGGT-OMEGA outputs。

   - full scene cache（完整 scene cache）。
   - 每个 candidate subset 的 hard subset cache。
   - 必需 tensors: `camera_tokens.pt`, `register_tokens.pt`, `depth.pt`, `depth_conf.pt`, `pose_enc.pt`。

4. 计算 labels。

   - subset-vs-full native depth consistency。
   - subset-vs-full pose rotation/center consistency。
   - derived point-map consistency。
   - 可选 sensor depth sanity metrics。

5. 训练 baselines。

   - parameter-free mean-pooled register cosine。
   - pooled MLP readout。
   - attention RegisterReadoutHead。

6. 用 scene-held-out split 验证。

   - 计算 scene-wise Spearman/Pearson。
   - 计算 expected-sign count。
   - 计算 best-score vs best-quality match。
   - 将每个 trained readout 与 mean pooling 对比。

7. 做决策。

   - 如果 readout 通过 gate，冻结 checkpoint 并 promotion 到 `0004_stage2_fixed_k_selector_training`。
   - 如果没有 readout 通过，则保留 mean pooling 作为 selector baseline objective，不依赖 trained readout。

## Hard-Label Pilot 流程

`train500_full16` warmup 之后，运行 `hardlabel100_full100_80`：

1. 选择 50 个 WildRGBD 和 50 个 DL3DV scenes，排除 LTM30 validation scenes。
2. 使用 full-view frames：

   - WildRGBD: 100 frames。
   - DL3DV: 80 frames。

3. 每个 scene 生成 12 个 hard subsets：

   - 5 x `random20`。
   - 3 x `random50`。
   - `uniform20`, `uniform50`。
   - `contiguous20_seed000`, `contiguous50_seed000`。

4. 为 full 和所有 hard subsets 缓存带 depth/pose 的 VGGT-OMEGA 输出。
5. 计算 subset-vs-full native geometry labels。
6. 用 same-scene good/bad subsets 的 pairwise ranking 训练 pooled readout。
7. 在 LTM30 hard subset metrics 上验证。
8. 如果 pooled readout 仍失败，保留 hard labels 并训练 2-layer attention readout。

实现命令：

```bash
/home/m/project/ltm/vggt-omega/.venv/bin/python scripts/prepare_stage2_readout_hardlabel100.py
```

## Attention Multi-Metric 跟进

复用 `hardlabel100_pooled_mlp_full100_80` 已完成的 hard-label cache 和 labels；不重新跑 VGGT cache。

目标命令：

```bash
tmux new -s readout0003_attention_multimetric
/home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_readout_attention_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv \
  --run-dir runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric \
  --train-devices cuda:0,cuda:1 \
  --epochs 30 \
  --batch-size 16 \
  --pairs-per-scene-metric 24 \
  --num-workers 4
```

Primary evaluation 是 metric-head LTM30 expected alignment。Embedding-cosine expected alignment 作为 secondary diagnostic 报告。

完成结果：best metric-head expected alignment 在 epoch 25 达到 `0.5657`，略高于 pooled hard-label readout `0.5594`，但低于严格 promotion gate。

## Ratio-20 / Large-Margin Attention Ablation 消融

复用已完成的 `hardlabel100` cache 和 labels，但只训练 20% subset rows，并移除 near-tie metric pairs。

目标命令：

```bash
tmux new -s readout0003_ratio20_margin
/home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_readout_attention_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv \
  --run-dir runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric_ratio20_margin \
  --train-devices cuda:0,cuda:1 \
  --epochs 30 \
  --batch-size 16 \
  --pairs-per-scene-metric 24 \
  --method-contains 20 \
  --min-metric-margin-fraction 0.25 \
  --num-workers 4
```

训练前预期 pair summary：

- label rows: 100 scenes 共 `700`。
- total metric pairs 数量: `3,255`。
- pair counts: pose rotation `1,016`，point-map RMSE `1,108`，depth log RMSE `1,131`。

完成结果：

- Best metric-head expected alignment: epoch 7 达到 `0.3860`。
- Final metric-head expected alignment: `0.3244`。
- Best embedding diagnostic expected alignment: epoch 16 达到 `0.5759`，但由于 checkpointing 跟随 metric-head score，没有保留为 `best.pt`。
- 决策: 不 promotion；将其作为 negative multi-metric 20%-only/margin ablation。

Single-target ablations 可以用一个 metric 复用同一命令：

```bash
--metrics pose_rotation_mean_deg
--metrics pointmap_rmse_norm
--metrics depth_log_rmse
```

Depth-only 完成结果：

- 运行目录: `runs/0003_stage2_readout_calibration/hardlabel100_attention_depth_ratio20_margin/`
- Best depth-head expected alignment: epoch 20 达到 `0.4248`。
- Final depth-head expected alignment: `0.3010`。
- Best embedding diagnostic expected alignment: `0.5543`。
- 决策: 不在这个 20%-only/margin setup 下继续 pose-only/point-only。

## All-Ratio Single-Target Attention Ablations 消融

复用全部 `hardlabel100` rows，每次只训练一个 metric：

```bash
/home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_readout_attention_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv \
  --run-dir runs/0003_stage2_readout_calibration/hardlabel100_attention_<target>_allratio_single \
  --train-devices cuda:0,cuda:1 \
  --epochs 30 \
  --batch-size 16 \
  --pairs-per-scene-metric 24 \
  --metrics <metric> \
  --num-workers 4
```

完成结果：

| 目标 | Run dir suffix | Best head | Best embedding | 决策 |
|---|---|---:|---:|---|
| `pose_rotation_mean_deg` | `pose_allratio_single` | 0.5524 | 0.6495 | embedding 有希望 |
| `pointmap_rmse_norm` | `pointmap_allratio_single` | 0.5333 | 0.6476 | embedding 有希望 |
| `depth_log_rmse` | `depth_allratio_single` | 0.5067 | 0.6019 | embedding 有希望，head 偏弱 |

下一步：在使用该信号做 promotion decision 前，增加显式 `best_head.pt` 和 `best_embedding.pt` checkpointing。

Checkpointing 更新：

- `best.pt`: compatibility path，与 metric-head best 相同。
- `best_head.pt`: 显式 metric-head best。
- `best_embedding.pt`: 显式 embedding diagnostic best。
- `summary.json`: 包含 `best_head_expected_alignment` 和 `best_embedding_expected_alignment`。

Checkpoint rerun 命令：

```bash
/home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_readout_attention_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv \
  --run-dir runs/0003_stage2_readout_calibration/hardlabel100_attention_<target>_allratio_single_ckpt \
  --train-devices <devices> \
  --epochs 30 \
  --batch-size 16 \
  --pairs-per-scene-metric 24 \
  --metrics <metric> \
  --num-workers <workers>
```

Checkpoint rerun 完成结果：

| 目标 | Run dir suffix | Devices | Best head | Best embedding | Retained embedding checkpoint |
|---|---|---|---:|---:|---|
| `depth_log_rmse` | `depth_allratio_single_ckpt` | `cuda:0` | 0.4819 | 0.6000 | `best_embedding.pt` |
| `pointmap_rmse_norm` | `pointmap_allratio_single_ckpt` | `cuda:1` | 0.5333 | 0.6133 | `best_embedding.pt` |
| `pose_rotation_mean_deg` | `pose_allratio_single_ckpt` | `cuda:0,cuda:1` | 0.5219 | 0.5505 | `best_embedding.pt` |

兼容性检查后的 best retained checkpoint set：

| 目标 | Checkpoint | Expected alignment |
|---|---|---:|
| `pose_rotation_mean_deg` | `runs/0003_stage2_readout_calibration/hardlabel100_attention_pose_allratio_single/best.pt` | 0.6495 |
| `pointmap_rmse_norm` | `runs/0003_stage2_readout_calibration/hardlabel100_attention_pointmap_allratio_single_ckpt/best_embedding.pt` | 0.6133 |
| `depth_log_rmse` | `runs/0003_stage2_readout_calibration/hardlabel100_attention_depth_allratio_single_ckpt/best_embedding.pt` | 0.6000 |
| 均值 | n/a | 0.6210 |

决策：在此停止 readout 分支。retained per-target embedding set 刚好超过 mean pooling + `0.10` 的严格目标，但它不是 single unified readout。`0004` 继续保留 mean-pooled register cosine 作为保守 single-objective fallback；只有当 `0004` 明确支持 metric-specific readout losses，或先完成 follow-up combination evaluation 时，才使用这些 retained embedding checkpoints。

```bash
tmux new -s readout0003_hardlabel100
/home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_readout_hardlabel_training.py \
  --manifest docs/experiments/0003_stage2_readout_calibration/hardlabel100_manifest.json \
  --cache-root caches/vggt_omega/0003_stage2_readout_calibration/hardlabel100_full100_80_images512 \
  --run-dir runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80 \
  --cache-devices cuda:0,cuda:1 \
  --train-devices cuda:0,cuda:1 \
  --epochs 40 \
  --batch-size 12 \
  --pairs-per-scene 48
```

## 未来命令

这些是 implementation phase 的目标命令形状；在对应 CLI/config 存在前不应运行。

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage readout-build-dataset \
  --config configs/experiments/0003_stage2_readout_calibration.yaml
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage readout-cache-vggt \
  --config configs/experiments/0003_stage2_readout_calibration.yaml
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage readout-train \
  --config configs/experiments/0003_stage2_readout_calibration.yaml
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage readout-eval \
  --config configs/experiments/0003_stage2_readout_calibration.yaml \
  --checkpoint <readout_checkpoint>
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage record-run \
  --experiment 0003_stage2_readout_calibration \
  --stage stage2 \
  --method attention_readout \
  --dataset <scene_split_name> \
  --config configs/experiments/0003_stage2_readout_calibration.yaml \
  --notes "scene-held-out readout calibration"
```

## 必需产物

- scene split manifest。
- subset manifest，包含 generation method、ratio、seed、selected image list。
- full 和 subset runs 的 VGGT-OMEGA cache manifest。
- 每个 subset 的 native geometry labels。
- readout checkpoints。
- training curves。
- correlation、sign 和 best-match metrics 的 evaluation CSVs。
- 如果 gate 通过，则记录给 `0004` 的 frozen readout promotion note。

## 检查项

- 除非 split 文档明确变更，否则 LTM30 validation scenes 不用于 readout training。
- `0004` 的 selector training/evaluation scenes 不能悄悄复用 readout validation/test。
- `R` register count 和 `C` token dim 必须从 cache manifest/tensor shape 读取，不能 hard-code。
- 所有 correlations 先在每个 scene 内计算，再跨 scenes 汇总。
- 对 lower-is-better native errors，期望它们与 readout score/similarity 呈负相关。
- Direct sensor pose metrics 只作为 diagnostics 报告，不作为 gate metrics。
- 除非记录了精确 checkpoint path 和 validation summary，否则 `0004` 不使用该 readout checkpoint。
