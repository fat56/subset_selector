# 结果

## 运行: `main_v1_meanpool_selector`

- 状态: 已完成。
- 配置: `configs/experiments/0004_stage2_fixed_k_selector_training.yaml`
- Manifest 文件: `docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_manifest.json`
- 运行目录: `runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/`
- Cache 根目录: `caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512/`
- Cache 策略: 只保留 compact `selector_features.pt`；不保存 `depth.pt`、`depth_conf.pt` 或 dense pointmap cache。
- Selector 目标: full-scene mean-pooled register cosine。
- 预算: `20%` ratio，`K = round(0.20 * N)`，至少选择 1 帧。
- Cache 设备: `cuda:0,cuda:1`
- 训练设备: `cuda:0,cuda:1`
- 训练 epochs / steps: `20` / `2160`
- 训练耗时: feature cache 完成后 `94.94` 秒。
- Feature cache 结果: `2138/2138` scenes 成功，`0` 失败。

## 数据集

| 数据集 | 入选 scenes | 备注 |
|---|---:|---|
| `bridgedata_v2` | 1000 | `data/processed/bridgedata_v2` |
| `nyuv2` | 549 | `data/processed/nyuv2` |
| `tartanair` | 163 | `data/processed/tartanair` |
| `bonn` | 26 | `data/processed/bonn` |
| `yifei_scannetv2_hf` | 400 | `data/raw/ltm_datasets/yifei_scannetv2_hf` |

| Split | Scenes |
|---|---:|
| train | 1728 |
| val | 205 |
| test | 205 |

总计: `2138` scenes，`105204` sampled frames。

## 训练诊断

| Run ID | Epoch | Val soft cosine | Val hard proxy cosine | Soft-hard gap | 备注 |
|---|---:|---:|---:|---:|---|
| `best_hard_proxy.pt` | 18 | 0.992365 | 0.969982 | 0.022383 | 最佳 val hard proxy |
| `best_soft.pt` | 20 | 0.993935 | 0.965243 | 0.028691 | 与 `last.pt` 相同 |
| `last.pt` | 20 | 0.993935 | 0.965243 | 0.028691 | 最后一轮 |

## Proxy baseline 对照

对 val split 的 `205` scenes，用相同 cached `register_mean/full_embedding` 计算 20% hard proxy cosine。这个对照不重新跑 VGGT，只用于判断当前 mean-register proxy objective 是否有选择判别力。

| 方法 | Mean hard proxy cosine | 中位数 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| 全量 frames | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| uniform stride | 0.999042 | 0.999463 | 0.993098 | 0.999987 |
| random 20%, 5 seeds | 0.993221 | 0.996456 | 0.904004 | 0.999998 |
| learned topK, `best_hard_proxy.pt` | 0.969982 | 0.978341 | 0.849700 | 0.999973 |
| prefix 20% | 0.942779 | 0.950483 | 0.633729 | 0.999949 |

解读：

- `main_v1` 成功训练出了 selector checkpoint，但没有打过 uniform/random proxy baseline。
- 当前 mean-pooled register proxy 太容易被均匀采样拿到接近满分，因此它不是足够强的 selector promotion 指标。
- `best_hard_proxy.pt` 仍然比 prefix 20% 好，说明模型没有完全失效；但它学习到的排序没有超过简单 coverage baseline。
- 这个结果支持下一步改 objective，而不是直接进入下游 3DGS 验算。

## 预检查

2026-06-08 已完成 `--cache-only --max-scenes 2` smoke cache：

- `cuda:0` 和 `cuda:1` 各缓存 1 个 scene，返回码均为 `0`。
- `selector_features.pt` 写入成功。
- 验证 tensor shape:
  - `frame_features`: `[N, 8193]`
  - `register_mean`: `[N, 2048]`
  - `full_embedding`: `[2048]`
  - dtype: `torch.float16`

## Checkpoints

| Checkpoint | 选择指标 | 状态 |
|---|---|---|
| `best_soft.pt` | val soft cosine | 已完成 |
| `best_hard_proxy.pt` | val hard proxy cosine | 已完成 |
| `last.pt` | 最后一轮 epoch | 已完成 |

## 决策

暂不把 `main_v1_meanpool_selector` 推进到 hard subset VGGT 或 FastGS/3DGS 验证。工程框架可用，但 objective/baseline 对照说明，仅靠 mean-pooled register cosine 作为 selector 训练目标太弱。推荐下一步切到 `main_v2` objective：加入 uniform/random/k-center baseline ranking，或从 `0003` hard-native labels / metric-specific learned readout evidence 蒸馏监督。

## 运行: `main_v2_baseline_rank_selector`

- 状态: 已完成。
- 配置: `configs/experiments/0004_stage2_fixed_k_selector_training_main_v2.yaml`
- Cache: 复用 `caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512/`
- 运行目录: `runs/0004_stage2_fixed_k_selector_training/main_v2_baseline_rank_selector/`
- Objective: baseline-aware rank objective + uniform target CE。
- 预算: `20%` ratio。
- 训练 epochs / steps: `20` / `2160`
- 训练耗时: `101.15` 秒。

Full manifest 上的 1-epoch smoke:

| Epoch | Val soft cosine | Val hard proxy cosine | Uniform proxy | Random proxy | Hard - uniform | Hard - random |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.999021 | 0.998906 | 0.999042 | 0.993662 | -0.000137 | 0.005243 |

解读：smoke 已经恢复到接近 uniform 的 coverage，并在 cache-only proxy 上超过 random。仍需完整 20-epoch run 判断它是否能超过 uniform，而不是只模仿 uniform。

完整 run 的 checkpoints:

| Checkpoint | Epoch | Val hard proxy | Uniform proxy | Random proxy | Hard - uniform | Hard - random | Soft-hard gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `best_margin.pt` | 16 | 0.998955 | 0.999042 | 0.993662 | -0.000087 | 0.005293 | 0.000072 |
| `best_hard_proxy.pt` | 16 | 0.998955 | 0.999042 | 0.993662 | -0.000087 | 0.005293 | 0.000072 |
| `best_soft.pt` | 3 | 0.998713 | 0.999042 | 0.993662 | -0.000329 | 0.005051 | 0.000326 |
| `last.pt` | 20 | 0.998856 | 0.999042 | 0.993662 | -0.000187 | 0.005193 | 0.000040 |

`best_margin.pt` 的 proxy eval，使用 5 个 random seeds:

| 方法 | Mean hard proxy cosine | 中位数 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| 全量 frames | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| uniform stride | 0.999042 | 0.999463 | 0.993098 | 0.999987 |
| learned topK, `best_margin.pt` | 0.998955 | 0.999463 | 0.989059 | 0.999987 |
| random 20%, 5 seeds | 0.993221 | 0.996456 | 0.904004 | 0.999998 |
| prefix 20% | 0.942779 | 0.950483 | 0.633729 | 0.999949 |

解读：

- `main_v2` 修复了 `main_v1` 的主要失败模式：learned topK 从 `0.969982` 提升到 `0.998955`，并在 val 上比 random 高 `+0.005293`。
- 它仍未超过 uniform stride：最佳 checkpoint 的 `hard_minus_uniform = -0.000087`。
- soft-hard gap 很小（`0.000072`），所以剩余问题不是 soft/hard mismatch，而是 mean-register proxy 强烈偏向 uniform coverage。
- 不应只凭这个 proxy 推进到 hard subset VGGT/3DGS。下一步应换更强 objective 或 evaluator：hard-native labels、register-k-center/ranking candidates，或 `0003` learned-readout auxiliary。

## 运行: `main_v3_hardnative_candidate_selector`

- 状态: 已完成；val 有小幅正信号，但 test 未通过。
- 配置: `configs/experiments/0004_stage2_fixed_k_selector_training_main_v3.yaml`
- Runner 脚本: `scripts/run_stage2_selector_hardnative_candidate_training.py`
- 源 labels 文件: `runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv`
- 源 jobs 文件: `runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json`
- 源 token cache: `caches/vggt_omega/0003_stage2_readout_calibration/hardlabel300_full100_80_images512/`
- Candidate 集合: `uniform20`、`random20_seed000-004`、`contiguous20_seed000`
- Metric: hard-native `target_error`，越低越好。

全部 300 个 labeled scenes 的 precheck:

| Split | uniform20 | random20 mean | random20 best-of-5 | contiguous20 | labeled candidate oracle |
|---|---:|---:|---:|---:|---:|
| train | -0.8582 | 0.6240 | -0.5722 | 5.5037 | -1.0555 |
| val | -1.0574 | 0.5989 | -0.6834 | 5.5981 | -1.1961 |
| test | -0.9917 | 0.6833 | -0.5293 | 5.4085 | -1.2233 |

Oracle winner 分布:

| Candidate family | Scenes |
|---|---:|
| `uniform20` | 217 |
| `random20` | 82 |
| `contiguous20` | 1 |

解读：`uniform20` 是很强的 hard-native baseline，但仍有可测 headroom：labeled oracle 在每个 split 上都低于 uniform，包括 test（`-1.2233` vs `-0.9917`）。

Smoke 设置：30 scenes，`candidate_set`，1 epoch，小型 2-layer/256 hidden model，单卡。

Smoke 结果:

| Split | Learned error | Uniform error | Random mean | Random best-of-5 | Oracle | Uniform - learned | Pairwise acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -0.6294 | -0.6422 | 0.5732 | -0.5983 | -0.8239 | -0.0128 | 0.7440 |
| val | 0.1603 | -0.2140 | 0.7135 | 0.1498 | -0.3873 | -0.3744 | 0.8226 |
| test | -1.2483 | -1.2483 | 0.2931 | -0.7226 | -1.2483 | 0.0000 | 0.6667 |

Smoke 解读：

- Data loading、image-list mask reconstruction、compact feature cache、training loss、checkpoint save/load 和 metric reporting 均正常。
- 1-epoch smoke 不是 promotion 结果；它故意设置得很小，不能判断 learned 是否超过 uniform。
- 即使在 smoke run 中，pairwise ranking signal 也能被模型读到：1 个 epoch 后 val pairwise accuracy 达到 `0.8226`。

完整 run:

- 模型: `candidate_set`
- Epochs / steps: `60` / `900`
- 训练耗时: feature cache 后 `55.83` 秒。
- Feature cache: `300` scenes，只保存 compact `frame_features`。
- 最佳 checkpoint: `best_uniform_improvement.pt`
- 最佳 epoch / step: `33` / `495`
- 最佳 val `uniform_minus_learned_error`: `+0.0101`

最佳 checkpoint 指标:

| Split | Learned error | Uniform error | Random mean | Random best-of-5 | Oracle | Uniform - learned | Regret reduction | Win vs uniform | Oracle top1 | Pairwise acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | -1.0555 | -0.8582 | 0.6240 | -0.5722 | -1.0555 | +0.1973 | +0.1973 | 0.2833 | 1.0000 | 0.9682 |
| val | -1.0675 | -1.0574 | 0.5989 | -0.6834 | -1.1961 | +0.0101 | +0.0101 | 0.0333 | 0.8000 | 0.7854 |
| test | -0.7360 | -0.9917 | 0.6833 | -0.5293 | -1.2233 | -0.2557 | -0.2557 | 0.0333 | 0.6333 | 0.7765 |

完整 run 解读：

- `candidate_set` 几乎可以完美拟合 training scenes：train oracle top1 达到 `1.0000`。
- validation split 有真实但很薄的正窗口：`uniform_minus_learned_error = +0.0101`。
- held-out test split 明确失败：`uniform_minus_learned_error = -0.2557`。
- val/test 的 pairwise accuracy 仍非随机，但顶层 candidate selection 的 calibration 不足，不能安全偏离 `uniform20`。

同一 checkpoint 上的 uniform-fallback threshold sweep:

| Threshold selection | Split | Learned error | Uniform error | Oracle | Uniform - learned | Deviation rate |
|---|---|---:|---:|---:|---:|---:|
| best by val | val | -1.0847 | -1.0574 | -1.1961 | +0.0273 | 0.1000 |
| best by val | test | -0.7049 | -0.9917 | -1.2233 | -0.2869 | 0.2333 |
| test oracle scan | test | -0.9917 | -0.9917 | -1.2233 | 0.0000 | 0.0000 |

Threshold 解读：

- 保守 fallback 只在 `10%` 的 val scenes 上偏离 uniform，就能把 val 从 `+0.0101` 提升到 `+0.0273`。
- 同一 threshold 会让 test 变差；对 thresholds 做 test oracle scan 时，最佳 test 决策是永远不偏离 uniform。
- 因此当前 learned deviation signal 还不够可靠，不能用于 selector promotion。

Frame-score 对照:

- 运行目录: `runs/0004_stage2_fixed_k_selector_training/main_v3_frame_score_candidate_selector/`
- 模型: `frame_score`
- Epochs / steps: `60` / `900`
- 训练耗时: `55.50` 秒。
- 最佳 checkpoint: `best_uniform_improvement.pt`
- 最佳 val `uniform_minus_learned_error`: `0.0000`

| Split | Learned error | Uniform error | Oracle | Uniform - learned | Win vs uniform | Oracle top1 | Pairwise acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -0.8582 | -0.8582 | -1.0555 | 0.0000 | 0.0000 | 0.7167 | 0.6912 |
| val | -1.0574 | -1.0574 | -1.1961 | 0.0000 | 0.0000 | 0.7667 | 0.6820 |
| test | -0.9917 | -0.9917 | -1.2233 | 0.0000 | 0.0000 | 0.7333 | 0.7186 |

Frame-score 解读：

- `frame_score` 的 best checkpoint 等价于选择 `uniform20`，没有取得正提升。
- 训练过程中它多次偏离 uniform 后变差，因此逐帧平均分数不足以表达 fixed-K subset 的 coverage/diversity 关系。
- 这个结果支持保留 set-aware `candidate_set` 结构，而不是把 selector 简化成每帧独立打分。

小型 rank-only `candidate_set` 对照:

- 运行目录: `runs/0004_stage2_fixed_k_selector_training/main_v3_rankonly_small_candidate_selector_cuda0/`
- 模型: `candidate_set`，`2` layers，hidden `256`，dropout `0.2`
- Loss: rank-only, `ce_weight = 0.0`, `min_target_gap = 0.05`
- 训练设备: `cuda:0`
- Epochs / steps: `60` / `900`
- 训练耗时: `38.27` 秒。
- 最佳 val `uniform_minus_learned_error`: `+0.0162`

| Split | Learned error | Uniform error | Oracle | Uniform - learned | Deviation/win vs uniform | Oracle top1 | Pairwise acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -0.9864 | -0.8582 | -1.0555 | +0.1282 | 0.1375 | 0.8000 | 0.9384 |
| val | -1.0737 | -1.0574 | -1.1961 | +0.0162 | 0.0333 | 0.8000 | 0.8199 |
| test | -0.6826 | -0.9917 | -1.2233 | -0.3091 | 0.0000 | 0.6333 | 0.7884 |

Rank-only 解读：

- 降低容量并去掉 oracle CE 后，val 从 `+0.0101` 提升到 `+0.0162`，但 test 更差。
- 这个 run 说明问题不是 4-layer 过大这么简单；模型仍然学到了 scene-specific deviation，但没有学到稳定的 held-out calibration。
- 双卡 `DataParallel` 对这个小模型曾在 initial eval 触发 CUDA kernel error，单卡 `cuda:0` 复跑成功；该问题记录为 runtime quirk，不影响结果判断。

Post-hoc gate/ensemble 扫描:

- 脚本: `scripts/evaluate_stage2_selector_hardnative_gate.py`
- 输出: `runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector/ensemble_gate_scan.json`
- 输入: 4-layer `candidate_set` best checkpoint + 2-layer rank-only checkpoint。
- 扫描规则: single-model uniform fallback 和 two-model agreement fallback。

| Gate 选择 | Split | Learned error | Uniform error | Oracle | Uniform - learned | Deviation rate | Win vs uniform | Oracle top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| best by val, all thresholds | val | -1.0847 | -1.0574 | -1.1961 | +0.0273 | 0.1000 | 0.0667 | 0.8000 |
| best by val, all thresholds | test | -0.7049 | -0.9917 | -1.2233 | -0.2869 | 0.2333 | 0.0333 | 0.6000 |
| best by val, positive-margin only | val | -1.0737 | -1.0574 | -1.1961 | +0.0162 | 0.0667 | 0.0333 | 0.8000 |
| best by val, positive-margin only | test | -0.6826 | -0.9917 | -1.2233 | -0.3091 | 0.1667 | 0.0000 | 0.6333 |
| best by test oracle scan | val | -1.0574 | -1.0574 | -1.1961 | 0.0000 | 0.0000 | 0.0000 | 0.7667 |
| best by test oracle scan | test | -0.9917 | -0.9917 | -1.2233 | 0.0000 | 0.0000 | 0.0000 | 0.7333 |

Gate 解读：

- 按 val 自动选择 gate，最高 val 提升可到 `+0.0273`，但 test 仍明显为负。
- 限制为 positive-margin gate 后，结果退化到 rank-only checkpoint 本身：val `+0.0162`，test `-0.3091`。
- 对 test 做 oracle threshold scan 时，最优规则是完全不偏离 uniform，即 `uniform_minus_learned_error = 0.0000`。
- 因此现有 hardlabel300 / 7-candidate 训练路线没有得到可 promotion 的 selector；当前最可靠 selector 仍是 `uniform20` fallback。

## 当前决策

暂不把 0004 selector 推进到 hard subset VGGT rerun 或下游 3DGS/FastGS validation。

这轮真正有价值的结果是明确了 negative boundary：`candidate_set` 能拟合 hard-native candidate ranking，也能找到小幅 validation signal，但这个信号无法通过 held-out test。下一次训练前，0004 需要更大的 hard-native label set 和更丰富的 candidates：更多 scenes、更多 random seeds、uniform jitter、register/DINO k-center，以及显式 calibrated “safe to deviate from uniform” objective。
