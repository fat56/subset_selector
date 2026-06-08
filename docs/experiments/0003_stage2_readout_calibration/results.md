# 结果

## 运行: `train500_pooled_mlp_full16`

- 日期: 2026-06-07
- 训练 scenes: 500
- 训练来源: 250 WildRGBD + 250 DL3DV
- 每个 scene 的 full-view frames: 16
- Cached training images 数量: 8,000
- Cache devices（缓存设备）: `cuda:0,cuda:1`
- Train device（训练设备）: `cuda:0`
- 模型: pooled MLP readout
- Objective: full-token online subset-mask warmup，包含 positive cosine、symmetric InfoNCE 和 uniform-vs-contiguous ranking。
- Validation（验证设置）: LTM30 hard subset validation，30 scenes x 6 subsets = 180 rows。
- 运行目录: `runs/0003_stage2_readout_calibration/train500_pooled_mlp_full16/`

这是一个 warmup/calibration baseline。它没有使用 500 个 training scenes 的 hard native labels；LTM30 hard native metrics 只用于 validation。

## 输出文件

- `best.pt`: 按 LTM30 primary expected alignment 选择的 best checkpoint。
- `last.pt`: final checkpoint（最终 checkpoint）。
- `summary.json`: run-level summary（运行摘要）。
- `best_eval/summary.json`: best-checkpoint LTM30 validation summary。
- `best_eval/ltm30_readout_scores.csv`: best-checkpoint 的 per-subset readout scores。
- `ltm30_readout_scores.csv`: final-checkpoint readout scores。
- `training_history.json`: train-step logs（训练日志）。

## LTM30 验证

native geometry errors 越低，readout/register score 应越高，因此 Spearman 应为负。下表的 `Expected alignment` 是 `-mean_spearman`，越高越好。

| 方法 | pose rotation rho | point-map RMSE rho | depth log RMSE rho | Mean rho | Expected alignment |
|---|---:|---:|---:|---:|---:|
| mean-pooled register baseline（基线） | -0.5429 | -0.5181 | -0.4990 | -0.5200 | 0.5200 |
| pooled readout best checkpoint（最佳） | -0.5390 | -0.5581 | -0.4895 | -0.5289 | 0.5289 |
| pooled readout final checkpoint（最终） | -0.5219 | -0.5295 | -0.5010 | -0.5175 | 0.5175 |

Best checkpoint（最佳 checkpoint）：

- checkpoint 路径: `runs/0003_stage2_readout_calibration/train500_pooled_mlp_full16/best.pt`
- epoch（轮次）: 15
- step（步数）: 465
- primary expected alignment: 0.5289

## 解读

train500 pooled readout warmup 训练稳定，也产生了真实 validation signal，但还没有通过预设 readout gate。

正向信号：

- best checkpoint 相比 mean pooling 略微提升 mean primary expected alignment：`0.5289` vs `0.5200`。
- point-map RMSE correlation 明显提升：`-0.5581` vs `-0.5181`。
- training loop、dual-GPU cache path、LTM30 validation 和 checkpoint flow 都已跑通。

局限：

- 平均提升只有 `+0.0089`，远低于预设 `+0.10` gate。
- Pose rotation 略差于 mean pooling。
- Depth log RMSE 略差于 mean pooling。
- 该 run 使用 online subset-mask warmup，而不是 train500 的 hard native labels，因此不能视为最终 calibrated readout。

## 决策

暂不将这个 pooled readout checkpoint promotion 为 `0004` 的锁定 selector objective。

建议下一步：为较小的 held-out training subset 生成 hard subset native labels，然后训练 pairwise-ranking pooled readout 或 2-layer attention readout。`0004` 当前仍以 mean-pooled register cosine 作为 fallback objective。

## 运行: `hardlabel100_pooled_mlp_full100_80`

- 日期: 2026-06-07
- 训练 scenes: 100
- 训练来源: 50 WildRGBD + 50 DL3DV
- 每个 scene 的 full-view frames: WildRGBD 100, DL3DV 80
- 每个 scene 的 hard subsets: 12
- VGGT cache jobs: 1,300/1,300 成功
- Hard native label rows 数量: 1,200
- Cache devices（缓存设备）: `cuda:0,cuda:1`
- Train devices（训练设备）: `cuda:0,cuda:1`
- 模型: pooled MLP readout
- Objective: same-scene hard subsets 上的 pairwise ranking，并加入 positive full/subset cosine 和 symmetric InfoNCE auxiliary losses。
- Validation（验证设置）: LTM30 hard subset validation，30 scenes x 6 subsets = 180 rows。
- 运行目录: `runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80/`

这个 run 使用实际 hard native labels，而不是 online masks。每个 subset 都重新跑 VGGT-OMEGA，并与 full-view cache 中同一批图像的 `pose_rotation_mean_deg`、`pointmap_rmse_norm` 和 `depth_log_rmse` 对比。

## Hard-Label 输出

- `hardlabel_native_metrics.csv`: 1,200 个 hard subsets 的 subset-vs-full VGGT-native metrics。
- `hardlabel_train_labels.csv`: per-scene z-scored training targets 和 token paths。
- `hardlabel_summary.json`: label counts 和 target range。
- `best.pt`: 按 LTM30 primary expected alignment 选择的 best checkpoint。
- `best_eval/summary.json`: best-checkpoint LTM30 validation summary。
- `best_eval/ltm30_readout_scores.csv`: best-checkpoint 的 per-subset readout scores。
- `last.pt`: final checkpoint（最终 checkpoint）。
- `summary.json`: run-level summary（运行摘要）。
- `training_history.json`: train-step logs。

## Hard-Label LTM30 验证

native geometry errors 越低，readout/register score 应越高，因此 Spearman 应为负。下表的 `Expected alignment` 是 `-mean_spearman`，越高越好。

| 方法 | pose rotation rho | point-map RMSE rho | depth log RMSE rho | Mean rho | Expected alignment |
|---|---:|---:|---:|---:|---:|
| mean-pooled register baseline | -0.5429 | -0.5181 | -0.4990 | -0.5200 | 0.5200 |
| `train500_full16` pooled best | -0.5390 | -0.5581 | -0.4895 | -0.5289 | 0.5289 |
| `hardlabel100` pooled best | -0.5048 | -0.5962 | -0.5771 | -0.5594 | 0.5594 |
| `hardlabel100` pooled final | -0.4724 | -0.5619 | -0.5333 | -0.5225 | 0.5225 |

Best checkpoint（最佳 checkpoint）：

- checkpoint 路径: `runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80/best.pt`
- epoch（轮次）: 35
- step（步数）: 14,000
- primary expected alignment: 0.5594
- labels 生成后的训练耗时: 2,802.3 秒

## Hard-Label 解读

hard-label pilot 相比 warmup 有实际提升，但还不足以把 pooled MLP readout promotion 为锁定的 Stage 2 selector objective。

正向信号：

- Mean primary expected alignment 从 mean-pooled register baseline `0.5200` 提升到 `0.5594`。
- 相比 warmup best checkpoint 的提升更明显：`+0.0305`。
- Point-map RMSE 和 depth log RMSE correlations 相比 mean pooling 明显提升。
- 完整 hard-label pipeline 已端到端跑通：dual-GPU VGGT cache、native label generation、dual-device readout training、best-checkpoint validation。

局限：

- 相比 mean pooling 的平均提升为 `+0.0394`，低于预设 `+0.10` gate。
- Pose rotation correlation 从 `-0.5429` 退化到 `-0.5048`；模型似乎更容易学习 point/depth consistency，而不是 pose rotation。
- final checkpoint 差于 best checkpoint，因此 checkpoint selection 很重要，并且仍可能 over-training。
- 即使 hard labels 有效，pooled MLP architecture 对 camera/register token structure 可能仍然过于有损。

## Hard-Label 决策

暂不把 pooled MLP hard-label checkpoint promotion 为最终 `0004` selector objective。

建议下一步：保留这份 hard-label dataset，并训练 structured attention readout；或在决定是否冻结 readout 前扩展 hard-label scenes/subset diversity。Mean-pooled register cosine 仍作为 fallback selector proxy。

## 运行: `hardlabel100_attention_multimetric`

- 状态: 已完成。
- Train labels: 复用 `hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv`。
- VGGT cache: 已复用；没有重新生成 VGGT cache。
- 模型: cross-attention readout，包含 scene query 和三个 metric queries。
- Metric heads: `pose_rotation_mean_deg`, `pointmap_rmse_norm`, `depth_log_rmse`。
- Objective: per-metric pairwise ranking、full-view score anchor、低权重 embedding alignment 和 InfoNCE。
- Validation: LTM30 hard subset validation，以 metric-head expected alignment 为 primary score。
- Train devices（训练设备）: `cuda:0,cuda:1`
- Epochs（轮次）: 30
- Steps（步数）: 13,500
- Training time: 2,022.5 秒
- 运行目录: `runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric/`

这个 run 用于测试 pooled readout 的失败是否主要来自丢弃 camera/register token structure。

Validation summary（验证汇总）：

| 方法 | pose rotation rho | point-map RMSE rho | depth log RMSE rho | Mean rho | Expected alignment |
|---|---:|---:|---:|---:|---:|
| mean-pooled register baseline（基线） | -0.5429 | -0.5181 | -0.4990 | -0.5200 | 0.5200 |
| `hardlabel100` pooled best | -0.5048 | -0.5962 | -0.5771 | -0.5594 | 0.5594 |
| attention metric-head best | -0.5295 | -0.5505 | -0.6171 | -0.5657 | 0.5657 |
| attention metric-head final | -0.5029 | -0.5162 | -0.5752 | -0.5314 | 0.5314 |
| attention embedding-cosine best diagnostic | -0.5105 | -0.5219 | -0.5733 | -0.5352 | 0.5352 |

Best checkpoint（最佳 checkpoint）：

- checkpoint 路径: `runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric/best.pt`
- epoch（轮次）: 25
- step（步数）: 11,250
- primary metric-head expected alignment: 0.5657
- embedding-cosine diagnostic expected alignment: 0.5352

解读：

- Attention/multi-metric heads 相比 pooled hard-label readout 只有小幅提升：`0.5657` vs `0.5594`。
- 最强提升来自 depth log RMSE：`-0.6171`，优于 pooled 的 `-0.5771`。
- Pose rotation 收回了部分 pooled regression，但仍低于 mean-pooling baseline：`-0.5295` vs `-0.5429`。
- final checkpoint 再次低于 best checkpoint，说明 early stopping 必要。
- 结果未通过严格 `+0.10` promotion gate；architecture 本身不是主要瓶颈。

决策：

不把 attention readout promotion 为锁定的 `0004` selector objective。继续保留 mean-pooled register cosine 作为 selector fallback。下一步 readout 改进应优先考虑更多/更丰富的 hard labels 或 target audit，而不是继续小改 head。

## 运行: `hardlabel100_attention_multimetric_ratio20_margin`

- 状态: 已完成。
- Train labels: 复用 `hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv`。
- VGGT cache: 已复用；没有重新生成 VGGT cache。
- 模型: 与 `hardlabel100_attention_multimetric` 相同的 cross-attention multi-metric readout。
- Training filter: 只保留名称包含 `20` 的 methods。
- Pair filter: same-scene metric margin 至少为该 scene/metric range 的 `0.25`。
- 过滤后 label rows: 100 scenes 共 700 rows。
- 过滤后 training pairs: 3,255 metric pairs。
- Train devices（训练设备）: `cuda:0,cuda:1`
- Epochs（轮次）: 30
- Steps（步数）: 6,090
- Training time: 682.7 秒
- 运行目录: `runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric_ratio20_margin/`

这个 run 用于测试前一个 attention 结果是否受两类因素影响：训练混合了 20/50/80/100% subsets，而 LTM30 validation 只有 20% subset；以及 noisy near-tie pseudo-label pairs。

Validation summary（验证汇总）：

| 方法 | pose rotation rho | point-map RMSE rho | depth log RMSE rho | Mean rho | Expected alignment |
|---|---:|---:|---:|---:|---:|
| mean-pooled register baseline（基线） | -0.5429 | -0.5181 | -0.4990 | -0.5200 | 0.5200 |
| attention metric-head best | -0.5295 | -0.5505 | -0.6171 | -0.5657 | 0.5657 |
| ratio20/margin metric-head best | -0.3676 | -0.3581 | -0.4324 | -0.3860 | 0.3860 |
| ratio20/margin metric-head final | -0.3295 | -0.2895 | -0.3543 | -0.3244 | 0.3244 |
| ratio20/margin embedding diagnostic peak | -0.5410 | -0.5886 | -0.5981 | -0.5759 | 0.5759 |
| ratio20/margin embedding diagnostic final | -0.5619 | -0.5695 | -0.5600 | -0.5638 | 0.5638 |

Best metric-head checkpoint（最佳 metric-head checkpoint）：

- checkpoint 路径: `runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric_ratio20_margin/best.pt`
- epoch（轮次）: 7
- primary metric-head expected alignment: 0.3860
- 该 checkpoint 的 embedding-cosine diagnostic: 0.5422

Embedding diagnostic 备注：

- best embedding diagnostic 出现在 epoch 16，expected alignment 为 `0.5759`。
- training loop 当时按 metric-head alignment 选择 checkpoint，因此 epoch-16 的 embedding-best model 没有保留为 `best.pt`。
- 这个 diagnostic 仍然能说明 objective/checkpoint selection 问题，但它不是可 promotion 的 checkpoint artifact。

解读：

- ratio-20/margin filtering 对 metric heads 是明确负结果：`0.3860` 远低于 mean-pooled baseline `0.5200` 和 prior attention best `0.5657`。
- embedding diagnostic 仍有竞争力，峰值略高于 prior attention head best，但这个信号不是当前 primary objective 选择的对象。
- 结果说明瓶颈不只是 subset-ratio mismatch 或 near-tie pair noise。multi-metric head objective 在收窄后的 20%-only label set 上可能本身更脆。
- 不启动更大的 20%-only multi-metric run。下一步有价值的 ablation 是 single-target training，或修改 checkpoint selection/objective 来保留 embedding-best models。

决策：

不 promotion 这个 checkpoint。把该 run 作为反对 20%-only multi-metric head training 加 `0.25` margin filter 的证据。如果继续本地探索，优先从 `depth_log_rmse` 的短 single-target ablation 开始，或重跑能按 embedding expected alignment 保存 checkpoint 的版本。

## 运行: `hardlabel100_attention_depth_ratio20_margin`

- 状态: 已完成。
- Train labels: 复用 `hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv`。
- VGGT cache: 已复用；没有重新生成 VGGT cache。
- 模型: cross-attention readout，带一个 metric head。
- 目标 metric: `depth_log_rmse`。
- Training filter: 只保留名称包含 `20` 的 methods。
- Pair filter: same-scene metric margin 至少为 depth metric range 的 `0.25`。
- 过滤后 label rows: 100 scenes 共 700 rows。
- 过滤后 training pairs: 1,131 depth pairs。
- Train devices（训练设备）: `cuda:0,cuda:1`
- Epochs（轮次）: 30
- Steps（步数）: 2,100
- Training time: 252.8 秒
- 运行目录: `runs/0003_stage2_readout_calibration/hardlabel100_attention_depth_ratio20_margin/`

这个 run 用于测试 ratio20/margin regression 是否由 multi-target head interference 导致。

Validation summary（验证汇总）：

| 方法 | depth log RMSE rho | Expected alignment |
|---|---:|---:|
| mean-pooled register baseline（基线） | -0.4990 | 0.4990 |
| attention metric-head best, all metrics | -0.6171 | 0.6171 |
| ratio20/margin metric-head best, all metrics | -0.4324 | 0.4324 |
| ratio20/margin depth-only head best | -0.4248 | 0.4248 |
| ratio20/margin depth-only head final | -0.3010 | 0.3010 |
| ratio20/margin depth-only embedding diagnostic peak | -0.5543 | 0.5543 |
| ratio20/margin depth-only embedding diagnostic final | -0.5048 | 0.5048 |

Best checkpoint（最佳 checkpoint）：

- checkpoint 路径: `runs/0003_stage2_readout_calibration/hardlabel100_attention_depth_ratio20_margin/best.pt`
- epoch（轮次）: 20
- depth-head expected alignment: 0.4248
- 该 checkpoint 的 embedding diagnostic: 0.5124

解读：

- Depth-only training 没有挽救 ratio20/margin metric-head setup；它的 best depth correlation 还略低于 multi-metric ratio20/margin 的 depth head。
- prior all-ratio attention model 在 depth 上仍明显更好（`0.6171` expected alignment），因此移除 higher-ratio subsets 很可能也移除了有用 supervision，而不只是清理噪声。
- embedding diagnostic 仍强于 metric head，但低于 all-ratio attention depth head。
- 不在这个 20%-only/margin recipe 下继续投入 pose-only/point-only。下一步有用改动应修改 objective/checkpointing，或回到 all-ratio labels 加 single-target heads。

决策：

不 promotion。完成 depth-only check 后停止当前 20%-only/margin 分支。

## Runs: all-ratio single-target attention ablations

- 状态: 已完成。
- Train labels: 复用 `hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv`。
- VGGT cache: 已复用；没有重新生成 VGGT cache。
- 模型: cross-attention readout，每个 run 只有一个 metric head。
- Training filter: 无；每个 scene 使用全部 12 种 subset methods。
- Pair filter: 除 same-scene positive metric margin 外无额外过滤。
- Label rows: 100 scenes 共 1,200 rows。
- Training pairs: 每个 target metric 2,400 pairs。

Run dirs（运行目录）：

- `runs/0003_stage2_readout_calibration/hardlabel100_attention_pose_allratio_single/`
- `runs/0003_stage2_readout_calibration/hardlabel100_attention_pointmap_allratio_single/`
- `runs/0003_stage2_readout_calibration/hardlabel100_attention_depth_allratio_single/`

这个 ablation 用于测试 prior multi-metric attention result 是否受 cross-target interference 限制，而不是受 architecture 或 data scale 限制。

Validation summary（验证汇总）：

| 目标 | Best head rho | Best head alignment | Best embedding rho | Best embedding alignment | Final head alignment | Final embedding alignment |
|---|---:|---:|---:|---:|---:|---:|
| `pose_rotation_mean_deg` | -0.5524 | 0.5524 | -0.6495 | 0.6495 | 0.4895 | 0.5924 |
| `pointmap_rmse_norm` | -0.5333 | 0.5333 | -0.6476 | 0.6476 | 0.3943 | 0.5143 |
| `depth_log_rmse` | -0.5067 | 0.5067 | -0.6019 | 0.6019 | 0.5067 | 0.6019 |
| 三个 best heads 均值 | -0.5308 | 0.5308 | n/a | n/a | n/a | n/a |
| 三个 best embeddings 均值 | n/a | n/a | -0.6330 | 0.6330 | n/a | n/a |

与 prior best 对比：

| 方法 | pose alignment | pointmap alignment | depth alignment | Mean |
|---|---:|---:|---:|---:|
| mean-pooled register baseline（基线） | 0.5429 | 0.5181 | 0.4990 | 0.5200 |
| `hardlabel100_attention_multimetric` heads | 0.5295 | 0.5505 | 0.6171 | 0.5657 |
| all-ratio single-target heads | 0.5524 | 0.5333 | 0.5067 | 0.5308 |
| all-ratio single-target embeddings | 0.6495 | 0.6476 | 0.6019 | 0.6330 |

解读：

- Single-target metric heads 平均上没有超过 prior multi-metric head。Depth 退化最明显，从 `0.6171` 降到 `0.5067`。
- Single-target embeddings 明显强于 heads。它们的 per-target best average 为 `0.6330`，超过原始 strict gate target 约 `0.6200`。
- 这还不是可 promotion 的 single checkpoint：三个 embedding scores 来自三个独立 single-target runs，而且当时 training loop 还没有稳定保留 `best_embedding.pt`。
- 这个结果改变了下一步优先级。有用信号不是“训练更好的 metric head”，而是“显式化 embedding checkpointing/objective，然后评估 embedding-selected checkpoints 是否泛化、能否组合”。

决策：

不把任何当前 single-target checkpoint promotion 为最终 `0004` objective。继续 checkpointing/objective 步骤：保存 `best_head.pt`、`best_embedding.pt` 和 metric-specific best summaries，然后重跑有希望的 all-ratio single-target 配置或 embedding-primary variant。

## Runs: all-ratio single-target embedding-checkpoint reruns

- 状态: 已完成。
- Train labels: 复用 `hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv`。
- VGGT cache: 已复用；没有重新生成 VGGT cache。
- 模型: 相同的 cross-attention single-target readout。
- Code change 已验证：attention training 现在会写出 `best.pt`、`best_head.pt` 和 `best_embedding.pt`。

Run dirs（运行目录）：

- `runs/0003_stage2_readout_calibration/hardlabel100_attention_depth_allratio_single_ckpt/`
- `runs/0003_stage2_readout_calibration/hardlabel100_attention_pointmap_allratio_single_ckpt/`
- `runs/0003_stage2_readout_calibration/hardlabel100_attention_pose_allratio_single_ckpt/`

Validation summary（验证汇总）：

| 目标 | Train devices | Best head epoch | Best head alignment | Best embedding epoch | Best embedding alignment | Final head alignment | Final embedding alignment | Retained embedding checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `depth_log_rmse` | `cuda:0` | 15 | 0.4819 | 26 | 0.6000 | 0.4229 | 0.4514 | `best_embedding.pt` |
| `pointmap_rmse_norm` | `cuda:1` | 4 | 0.5333 | 4 | 0.6133 | 0.4533 | 0.5010 | `best_embedding.pt` |
| `pose_rotation_mean_deg` | `cuda:0,cuda:1` | 30 | 0.5219 | 15 | 0.5505 | 0.5219 | 0.5010 | `best_embedding.pt` |

旧 run 兼容性检查：

| 目标 | 可用旧 checkpoint | Embedding alignment | 原因 |
|---|---|---:|---|
| `pose_rotation_mean_deg` | `hardlabel100_attention_pose_allratio_single/best.pt` | 0.6495 | 旧 run 的 best head 和 best embedding 都发生在 epoch 15 |
| `depth_log_rmse` | `hardlabel100_attention_depth_allratio_single/last.pt` 或新 `best_embedding.pt` | 0.6019 old, 0.6000 new | 旧 run 的 best/final embedding 作为 `last.pt` 保留；新 run 显式保留 embedding-best checkpoint |
| `pointmap_rmse_norm` | `hardlabel100_attention_pointmap_allratio_single_ckpt/best_embedding.pt` | 0.6133 | 旧 peak `0.6476` 没有保留；新显式 checkpoint 保留了本次 rerun 中最好的可复现实点 |

Best retained per-target embedding checkpoint set（保留的最佳 per-target embedding checkpoint 集合）：

| 目标 | Checkpoint | Expected alignment |
|---|---|---:|
| `pose_rotation_mean_deg` | `runs/0003_stage2_readout_calibration/hardlabel100_attention_pose_allratio_single/best.pt` | 0.6495 |
| `pointmap_rmse_norm` | `runs/0003_stage2_readout_calibration/hardlabel100_attention_pointmap_allratio_single_ckpt/best_embedding.pt` | 0.6133 |
| `depth_log_rmse` | `runs/0003_stage2_readout_calibration/hardlabel100_attention_depth_allratio_single_ckpt/best_embedding.pt` | 0.6000 |
| 均值 | n/a | 0.6210 |

解读：

- Explicit embedding checkpointing 可用，并已在三个 single-target reruns 上验证。
- retained metric-specific embedding checkpoint set 的 mean expected alignment 达到 `0.6210`，刚好高于原始 strict readout gate target：mean-pooled baseline `0.5200` + `0.10`。
- 这是当前四步 readout 分支的可接受停止点，因为它产出了具体、可检查的 checkpoint artifacts，并以 per-target embedding set 形式跨过 gate。
- 它不是 single unified readout checkpoint。若要直接 promotion 到 `0004`，需要 metric-specific selector losses，或先完成 combination/embedding-primary evaluation。
- pose `_ckpt` rerun 没有复现旧 pose embedding 高点；旧 pose `best.pt` 仍是更好的 retained pose artifact。

决策：

在这里停止当前 readout 分支，不再继续小型 head ablations。保留 mean-pooled register cosine 作为保守 single-objective fallback，并把 retained per-target embedding checkpoint set 作为未来 `0004` 设计中最好的 learned readout 证据。

## Runs: hardlabel300 attention multimetric scale-up

- 状态: 2-layer run 已完成；4-layer run 正在训练。
- Train labels: 新生成 `hardlabel300_full100_80`。
- 训练 scenes: 300，包含 150 WildRGBD + 150 DL3DV。
- 每个 scene 的 full-view frames: WildRGBD 100, DL3DV 80。
- 每个 scene 的 hard subsets: 12。
- VGGT cache jobs: 3,900/3,900 成功。
- Hard native label rows 数量: 3,600。
- Train devices（训练设备）: `cuda:0,cuda:1`。
- Primary label metrics: `pose_rotation_mean_deg`, `pointmap_rmse_norm`, `depth_log_rmse`。

Run dirs（运行目录）：

- `runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/`
- `runs/0003_stage2_readout_calibration/hardlabel300_attention_multimetric_2layer/`
- `runs/0003_stage2_readout_calibration/hardlabel300_attention_multimetric_4layer/`

这个 scale-up 用来测试前面 `hardlabel100_attention_multimetric` 的瓶颈是否主要来自 hard-label 数据量和训练轮次，而不是只来自 readout 架构。2-layer run 使用相同 cross-attention multi-metric readout，但把训练 labels 从 100 scenes / 1,200 rows 扩到 300 scenes / 3,600 rows，并跑 40 epochs。

Validation summary（2-layer 已完成）：

| 方法 | pose alignment | pointmap alignment | depth alignment | Mean expected alignment |
|---|---:|---:|---:|---:|
| mean-pooled register baseline（基线） | 0.5429 | 0.5181 | 0.4990 | 0.5200 |
| `hardlabel100_attention_multimetric` head best | 0.5295 | 0.5505 | 0.6171 | 0.5657 |
| hardlabel300 2-layer head best | 0.4895 | 0.5867 | 0.6190 | 0.5651 |
| hardlabel300 2-layer embedding best | 0.6019 | 0.5829 | 0.6343 | 0.6063 |
| hardlabel300 2-layer final embedding | 0.5143 | 0.6210 | 0.6590 | 0.5981 |

2-layer best checkpoints（最佳 checkpoints）：

- `best_head.pt`: epoch 20，mean primary expected alignment `0.5651`。
- `best_embedding.pt`: epoch 29，embedding mean primary expected alignment `0.6063`。
- `best.pt`: compatibility path，与 `best_head.pt` 相同。
- 训练步数: 54,000。
- 训练耗时: 8,195.6 秒。

解读：

- 扩大 hard-label 数据和训练轮次明显改善了 embedding diagnostic：best embedding 从 hardlabel100 multi-metric 的 `0.5352` 提升到 `0.6063`。
- 2-layer metric heads 没有超过 hardlabel100 attention head best：`0.5651` vs `0.5657`，说明 head objective 仍是瓶颈。
- `best_embedding.pt` 在三项 primary metrics 上都超过 mean-pooled register baseline，但 mean `0.6063` 仍低于严格 single-checkpoint gate target 约 `0.6200`。
- 最好点出现在 epoch 29；后续训练有回落，final head 只有 `0.4571`，final embedding 为 `0.5981`。因此 early stopping/checkpoint selection 仍必要。
- 当前 2-layer 结果不改变 `0004` 的保守 fallback 决策；4-layer run 继续用于测试更深 attention 是否能把 unified embedding 推过 `0.62`。

## 决策日志

- 2026-06-07: LTM30 显示强 VGGT-native subset-vs-full signal，而 Stage 1 appearance/sparse-geometry proxies 仍不足，因此创建本设计。
- 2026-06-07: 跑完 `train500_pooled_mlp_full16`；训练稳定，但没有通过 readout-improvement gate。
- 2026-06-07: 跑完 `hardlabel100_pooled_mlp_full100_80`；hard native labels 相比 warmup 明显改善 pooled readout，但仍未通过严格 `+0.10` gate。
- 2026-06-07: 跑完 `hardlabel100_attention_multimetric`；attention 和独立 metric heads 将 best alignment 小幅提升到 `0.5657`，仍低于严格 promotion gate。
- 2026-06-07: 跑完 `hardlabel100_attention_multimetric_ratio20_margin`；metric-head alignment 退化到 `0.3860`，embedding diagnostics 峰值为 `0.5759`，但没有保留 embedding-best checkpoint。
- 2026-06-07: 跑完 `hardlabel100_attention_depth_ratio20_margin`；depth-only head 仍表现不足（`0.4248`），因此不值得继续扩大 20%-only/margin 分支。
- 2026-06-08: 跑完 all-ratio single-target attention ablations；metric heads 仍一般（mean `0.5308`），但 single-target embeddings 很强（mean best alignment `0.6330`），因此下一步优先做 explicit embedding checkpointing。
- 2026-06-08: 添加 explicit `best_head.pt` 和 `best_embedding.pt` checkpointing，重跑 all-ratio single-target checkpoint checks，并以 retained per-target embedding checkpoint set 的 `0.6210` mean expected alignment 停止该分支。
- 2026-06-08: 扩大到 `hardlabel300_full100_80` 后跑完 2-layer attention multi-metric 40 epochs；best embedding 达到 `0.6063`，明显优于 hardlabel100 multi-metric embedding diagnostic，但仍未通过 single unified checkpoint 的 `0.62` 目标。4-layer 对照继续运行。
