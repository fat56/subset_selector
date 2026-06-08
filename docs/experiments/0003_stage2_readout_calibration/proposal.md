# Stage 2.0 Readout 校准

## 元数据

- 实验 ID: `0003_stage2_readout_calibration`
- 阶段: `stage2`
- 状态: all-ratio embedding-checkpoint follow-up 已完成；当前 readout 分支已停止
- 创建日期: 2026-06-07
- Config: 已用 `hardlabel100_full100_80` 实现
- 依赖: `0001_stage1_register_quality_gate`, `0002_ltm30_pose_depth_validation`
- 输入到: `0004_stage2_fixed_k_selector_training`

## 问题

在 Stage 1 的 mean-pooled register-token proxy 只得到中等信号之后，是否应该先训练或校准一个 lightweight readout head，把 VGGT-OMEGA 的 camera/register tokens 转成更可靠的场景级 3D understanding proxy，再把它冻结给 Stage 2 selector 使用？

## 当前证据

现有证据支持开始 Stage 2，但支持的是一个收窄后的 Stage 2.0：先做 readout calibration 和 VGGT-native geometry proxy，而不是直接进入最终 selector/FastGS/VLA 结论。

### Stage 1 没有证明什么

`0001_stage1_register_quality_gate` 中，mean-pooled register token 和 PSNR/SSIM/LPIPS 的 scene-wise 相关性偏弱；FastGS appearance quality 对 register token 的 3D understanding 能力不是最佳目标。

补跑的 sparse geometry proxy 也没有支持 mean-pooled register token 直接作为 selector training signal：对 `colmap_sparse_full_scene` 的 accuracy/completeness/Chamfer/F-score，mean Spearman 接近 0，best-method 一致率很低。

### LTM30 证明了什么

`0002_ltm30_pose_depth_validation` 在 30 个 pose/depth scene 上提供了更强的验证。mean-pooled register cosine 和 VGGT-native subset-vs-full consistency 有明确方向：

| 指标 | Mean Spearman | 符号 | Best Match |
|---|---:|---:|---:|
| `pose_rotation_mean_deg` | -0.5429 | 29/30 | 21/30 |
| `pointmap_rmse_norm` | -0.5181 | 28/30 | 22/30 |
| `depth_log_rmse` | -0.4990 | 28/30 | 25/30 |
| `depth_absrel_mean` | -0.4819 | 30/30 | 24/30 |
| `pose_center_rmse_norm` | -0.4857 | 26/30 | 19/30 |

因此，register tokens 确实含有可用于判断 subset 3D consistency 的信息；不足之处在于当前读法太粗糙。

### Hard-Label Pilot 结果

`hardlabel100_full100_80` 已完成：100 个训练 scene，1,300 个 VGGT cache jobs，1,200 个 subset-vs-full hard native labels。pairwise-ranking pooled MLP readout 的 best checkpoint 在 LTM30 held-out validation 上达到：

| 方法 | Expected alignment |
|---|---:|
| mean-pooled register baseline | 0.5200 |
| `train500_full16` warmup best | 0.5289 |
| `hardlabel100_full100_80` best | 0.5594 |
| `hardlabel100_attention_multimetric` best | 0.5657 |

该结果说明 hard native labels 确实比 weak online mask warmup 更有效；cross-attention/multi-metric head 也略优于 pooled MLP。但 best attention readout 相对 mean pooling 的提升为 `+0.0457`，仍未达到原定 `+0.10` readout gate。因此本轮 readout 暂不晋级为 `0004` 的锁定 selector objective；mean-pooled register cosine 仍保留为 fallback。下一步更合理的是扩大 hard-label scenes/subset diversity，或重新审计 VGGT-native pseudo-label 目标，而不是继续只调小 head。

### Attention Multi-Metric 跟进

下一步先不重新跑 VGGT，也不盲目扩大 hard-label 数据，而是复用 `hardlabel100_full100_80` 的 token cache 和 1,200 条 hard labels，训练一个 token-structure-aware readout：

```text
camera/register tokens
  -> Linear(2048, 512)
  -> frame embedding + token-slot embedding
  -> learned scene/metric queries cross-attend to all VGGT tokens
  -> scene embedding + per-metric quality heads
```

关键变化：

- 用 cross-attention queries 读取 token，而不是 pooled `mean/max/std`。
- 三个 metric head 分别学习 `pose_rotation_mean_deg`、`pointmap_rmse_norm`、`depth_log_rmse`，不再强行先混成一个 `target_error`。
- 训练目标是同一 scene 内、同一 metric 下的 pairwise ranking：低 native error subset 的 metric score 应高于高 native error subset。
- full-view cache 作为 anchor：full 的 metric score 应高于 subset。
- 低权重保留 embedding-to-full alignment 和 InfoNCE，用于稳定 scene embedding，但 gate 主要看 metric-head score。

本 follow-up 的判断标准：

- 如果 metric-head expected alignment 明显超过 `0.5594`，说明瓶颈主要是 pooled MLP 架构。
- 如果仍停在 `0.56` 左右，则应优先扩大 hard-label scenes/subset diversity 或重新审计 pseudo-label 目标，而不是继续调小 head。
- 若 pose head 仍弱而 point/depth head 强，说明 pose signal 需要独立 selector/readout 或单独 pose-target calibration。

结果：attention follow-up 的 best checkpoint 达到 `0.5657` metric-head expected alignment。它只比 pooled MLP 高 `+0.0063`，说明瓶颈不只是 pooled summarization。这个 readout 仍未通过严格 promotion gate。

### Ratio-20 / Large-Margin 跟进

下一组低成本 ablation 复用同一份 `hardlabel100_full100_80` cache 和 labels，但调整训练分布以更贴近 LTM30 validation：

- 只保留 20% subset methods：`random20_*`、`uniform20` 和 `contiguous20_*`。
- 只保留 same-scene pair candidates，并要求 metric margin 至少达到该 scene 对应 metric range 的 `0.25`。
- 保持 cross-attention multi-metric 架构和 LTM30 evaluation protocol。
- 通过 `--metrics` 传入单个 metric，复用同一套 CLI 做 single-target ablations。

这用于测试前一个 readout 是否被混合 subset ratios 和 noisy near-tie pairs 稀释。过滤后的训练集包含 100 个 scene、700 条 label rows 和 3,255 个 pairwise metric examples。

结果：metric-head objective 明显退化，best expected alignment 只有 `0.3860`。embedding diagnostic 峰值为 `0.5759`，但当时训练循环按 metric-head score 保存 checkpoint，因此没有保留 embedding-best checkpoint。depth-only single-target 检查也表现不佳，best depth-head expected alignment 为 `0.4248`。直接结论是：20%-only/margin filtering 不充分，甚至可能移除了有用的 all-ratio supervision；后续 ablation 应修改 objective/checkpoint selection，或回到 all-ratio labels 加 single-target heads。

### All-Ratio Single-Target 跟进

已经完成 `pose_rotation_mean_deg`、`pointmap_rmse_norm` 和 `depth_log_rmse` 三个 all-ratio single-target runs。Metric heads 仍然一般，mean best expected alignment 为 `0.5308`；但 single-target embeddings 明显更强：

| 目标 | Best head alignment | Best embedding alignment |
|---|---:|---:|
| `pose_rotation_mean_deg` | 0.5524 | 0.6495 |
| `pointmap_rmse_norm` | 0.5333 | 0.6476 |
| `depth_log_rmse` | 0.5067 | 0.6019 |
| 均值 | 0.5308 | 0.6330 |

如果把它看作 per-target embedding diagnostics，这已经超过原始 strict gate target；但它还不是单个可 promotion 的 checkpoint。当时下一步实现重点是显式保存 embedding-best checkpoint，并补上 embedding-primary objective/evaluation path。

checkpointing follow-up 增加了显式 `best_head.pt` 和 `best_embedding.pt` 输出，然后重跑 all-ratio single-target 配置。这些 reruns 验证了三个目标的 embedding-best checkpoint retention 都能工作。当前 best retained per-target embedding checkpoint set 为：

| 目标 | Checkpoint 来源 | Expected alignment |
|---|---|---:|
| `pose_rotation_mean_deg` | older `pose_allratio_single/best.pt` | 0.6495 |
| `pointmap_rmse_norm` | `pointmap_allratio_single_ckpt/best_embedding.pt` | 0.6133 |
| `depth_log_rmse` | `depth_allratio_single_ckpt/best_embedding.pt` | 0.6000 |
| 均值 | n/a | 0.6210 |

它刚好超过原始 strict target `0.5200 + 0.10`，但仅限于 metric-specific checkpoint set。这是当前分支可接受的停止点，不代表有一个 unified readout checkpoint 可以直接 promotion。对于 `0004`，除非 selector 设计明确支持 metric-specific readout losses，或先完成 embedding-combination evaluation，否则继续把 mean-pooled register cosine 作为保守的 single-objective fallback。

### External GT 注意事项

WildRGBD sensor depth 有弱但可用的方向性；direct sensor pose GT 目前不可靠，不应作为主训练目标。第一版 readout 应优先对齐 VGGT-native subset-vs-full depth、derived point-map、pose rotation consistency，并把 sensor depth 作为 sanity check。

## 假设

一个冻结 VGGT-OMEGA 之上的小型 RegisterReadoutHead，可以比 mean pooling 更稳定地把 camera/register tokens 读成场景级 embedding 或质量分数。该 readout 的输出距离或 score 应该和下列 VGGT-native errors 在 scene 内负相关：

- `pose_rotation_mean_deg`
- `pointmap_rmse_norm`
- `depth_log_rmse`
- `depth_absrel_mean`
- `pose_center_rmse_norm`

如果 readout calibration 成功，则 `0004_stage2_fixed_k_selector_training` 可以使用 frozen readout 作为 selector 的稳定 proxy loss。

## 范围

本实验只训练或校准 readout head，不训练 selector。

包括：

- 使用 frozen VGGT-OMEGA cache，不更新 VGGT-OMEGA。
- 训练 parameter-free mean pooling、small MLP readout、attention readout 三类读法的对照。
- 用 scene-held-out split 验证 readout distance/score 与 VGGT-native geometry metrics 的相关性。
- 评估 readout 是否比 mean pooling 更适合作为 selector loss。

不包括：

- 不训练 fixed-K selector；该内容顺延到 `0004_stage2_fixed_k_selector_training`。
- 不用 VLA/downstream task 作本阶段主指标。
- 不把 direct external pose GT 作为主目标，除非先完成 pose convention audit。
- 不端到端更新 VGGT-OMEGA。

## Readout 候选

### Baseline A: Mean-Pooled Register Cosine 基线

当前 Stage 1/LTM30 使用的 parameter-free baseline：

```text
register_tokens: [B, N, R, C]
z = mean(register_tokens, dim=[B, N, R])
z = L2Normalize(z)
similarity = cosine(z_subset, z_full)
```

优点是无训练、稳定、容易复现；缺点是丢失 frame order、camera token、register slot structure、coverage/redundancy 信息。

### Baseline B: Pooled MLP Readout 基线

```text
per-scene summary = concat(
    mean(camera_tokens),
    mean(register_tokens),
    max(register_tokens),
    std(register_tokens),
    optional scalar summaries
)
summary -> LayerNorm -> MLP -> z or quality score
```

预计参数量约 `1-3M`。适合作为第一个训练式 readout，能判断“少量可学习校准是否已经足够”。

### Candidate C: Attention RegisterReadoutHead 候选

VGGT-OMEGA cache 中 observed token shape 为：

```text
camera/register tokens: [B, N, R + 1, C]
C = 2048
R = 16 in current checkpoint cache, but implementation must read from manifest/checkpoint.
```

推荐结构：

```text
tokens
    -> reshape [B, N * (R + 1), C]
    -> Linear(C, 512)
    -> concat 1 learnable readout token
    -> 2 x Pre-LN TransformerEncoderBlock
    -> take readout token
    -> Linear(512, 256 or 512)
    -> L2 normalize
```

参数量估算：

| 版本 | 主要维度 | 约参数量 |
|---|---|---:|
| pooled MLP | pooled 2048-d summaries -> 256/512 | 1-3M |
| attention readout，2 blocks | 2048 -> 512，2 Transformer blocks | 7-8M |
| attention readout，4 blocks | 2048 -> 512，4 Transformer blocks | 13-14M |

这些参数量相对 VGGT-OMEGA 很小；训练成本主要来自 subset label/cache 生成，而不是 head 本身。

实现备注：当前 follow-up 使用 cross-attention 变体，而不是对所有 image tokens 做 full self-attention，因此显存大致随 query count x token count 增长，而不是随 token count 的平方增长。

## 训练目标

第一版推荐把 readout 训练成 ranking/calibration head，而不是只做 scene identity contrastive learning。

### Pairwise Ranking 目标

同一 scene 内，如果 subset A 的 native error 明显低于 subset B，则 readout 应给 A 更高质量或更接近 full 的 embedding：

```text
score(S) = -distance(readout(tokens_S), readout(tokens_full))
L_rank = max(0, margin - score(S_good) + score(S_bad))
```

`S_good` / `S_bad` 可由组合指标排序得到：

```text
target_error =
    w_rot * zscore(pose_rotation_mean_deg)
  + w_point * zscore(pointmap_rmse_norm)
  + w_depth * zscore(depth_log_rmse)
```

MVP 权重：

```text
w_rot = 1.0
w_point = 1.0
w_depth = 1.0
```

### Metric Regression 目标

辅助训练一个 scalar quality/error head：

```text
readout(tokens_subset, tokens_full) -> predicted_error
L_reg = SmoothL1(predicted_error, target_error)
```

这能让 readout 的分数更可解释，但不能单独替代 hard subset validation。

### Embedding Alignment 目标

训练 subset embedding 接近 full embedding：

```text
L_pos = 1 - cosine(z_subset, stopgrad(z_full))
```

仅使用 `L_pos` 可能会过度鼓励所有 subset 都靠近 full，无法区分好坏 subset。推荐只作为辅助项，与 ranking/regression 搭配。

## 数据需求

当前 LTM30 的 30 scenes x 6 subset = 180 subset pairs，更适合作为 held-out validation，不足以训练一个泛化 readout。

推荐数据规模：

| 用途 | Scenes | 每个 scene 的 subsets | Pairs |
|---|---:|---:|---:|
| smoke test | 50-100 | 10-20 | 500-2,000 |
| 可用的小 readout | 100-200 | 20-50 | 2,000-10,000 |
| 稳健 readout | 200-500 | 20-50 | 4,000-25,000 |

Split 必须按 scene 划分。不能让同一个 scene 的不同 subset 同时进入 train 和 validation。

子集生成建议：

- ratios: `10%`, `20%`, `30%`
- methods: random seeds、uniform stride、contiguous windows、farthest/k-center candidates、high-overlap low-diversity negatives
- full: 每个 scene 最多 200 张图，排除测试集或使用当前 LTM full split 口径

LTM30 当前 30 个 scene 建议保留为 validation set；如果需要训练，优先从 `/home/m/dataset/ltm_datasets` 中抽取额外 pose/depth scenes。

### 本地数据审计

2026-06-07 对 `data/raw/ltm_datasets -> /home/m/dataset/ltm_datasets` 做了一次只读扫描。当前已有数据足够构建较可靠的 readout calibration set，甚至不需要动用全部数据。

现有 scanner 可直接识别的候选如下：

| Dataset | 可用 scenes | Frames | Depth | Pose | 备注 |
|---|---:|---:|---|---|---|
| `wildrgbd_harrison` | 1,998 | 199,800 | sensor depth png | npz `camera_pose` | 每个 scene 100 帧；最适合作 readout 训练和 sensor-depth sanity。 |
| `DL3DV-ALL-480P` | 976 | 78,080 | COLMAP photometric depth | NeRF `transforms.json` c2w | 每个 scene 80 帧；适合补充室外/大场景与 SfM-style pseudo depth。 |
| `yifei_scannetv2_hf` | 1,510 | 479,891 | 本地副本无 depth | txt camera pose | pose-only；可做 VGGT-native pseudo-label 和 selector robustness，不适合作 sensor-depth sanity。 |
| `MegaDepth_v1` | 尚未统计 | n/a | format support 完成后应可用 | SfM-style | 本地占用约 203G，但当前 scanner 未接入；先不作为 MVP 依赖。 |

合计当前 scanner 可见 `4,484` 个 pose scene，`757,771` 帧；其中 `2,974` 个 scene 同时有 depth 和 pose。这个规模已经超过 robust readout 的建议规模 `200-500` scenes / `4k-25k` subset pairs。

因此第一版建议只抽一个可控子集：

- readout train: 从 WildRGBD + DL3DV 抽 `300-500` 个 scenes。
- readout validation: 保持当前 LTM30 held out，可选再加入 `50-100` 个额外 held-out scenes。
- optional pseudo-label expansion: 只有在 depth+pose readout baseline 可用后，再加入 ScanNet pose-only scenes。
- MegaDepth: 等 scanner/format audit 完成后再接入。

### 与 Stage 2 Selector 的 Split 策略

Readout calibration data 应与 selector evaluation data 分离，最好也与 selector training data 分离。

严格策略：

- `readout_train`: used only to train/calibrate the readout.
- `readout_val/test`: used only to choose readout checkpoint and threshold; current LTM30 should live here.
- `selector_train`: used to train the fixed-K selector in `0004`; no scene overlap with `readout_val/test`.
- `selector_val/test`: used for hard subset VGGT/FastGS/VLA validation; no scene overlap with readout or selector training.

如果数据稀缺，`readout_train` 与 `selector_train` 可以重叠，因为 readout 会在 selector training 前冻结；但这会削弱实验结论的可信度。考虑到当前 `ltm_datasets` 规模和用户额外约 500G 的数据预算，推荐默认避免重叠。

额外约 500G 数据更适合作 distribution buffer：

- 拿一部分做 `selector_train`，避免 selector 只学习 readout-calibration scenes；
- 保留干净的 final test slice，用于 Stage2 后的 hard VGGT/FastGS/VLA validation；
- 不要把额外数据全花在 readout 上，因为 readout 只需要几百个 scene，而 selector/generalization validation 更需要独立数据。

### 已选 Train500 MVP

第一个实际运行实验使用 [train500_manifest.json](train500_manifest.json)：

| 设置 | 值 |
|---|---:|
| WildRGBD scenes | 250 |
| DL3DV scenes | 250 |
| 训练 scenes 总数 | 500 |
| 每个 scene 的 full-view frames | 16 |
| cached training images 总数 | 8,000 |
| 排除的 LTM30 validation scenes | 30 |

选择策略：

- deterministic random seed: `20260607`
- scene sampling: balanced WildRGBD/DL3DV，排除当前 LTM30 validation scenes
- frame sampling: 对每个 scene 的 eligible frames 做 uniform downsample
- cache scope: 只缓存 full-view camera/register tokens
- training subsets: 从 full cached tokens 在线采样 masks，ratio 为 `0.25`、`0.5`、`0.75`
- validation: LTM30 hard subset VGGT-native metrics 只用于 validation

这是一个快速进入训练的 warmup/calibration run。它还不会为 500 个 training scenes 全部生成 hard native labels；等 full-cache readout baseline 测完后，再启用更重的 follow-up。

### Hard-Label Full-View Pilot 试点

`train500_full16` warmup 证明了 pipeline 可行，但没有通过 readout gate。下一步应把算力用在更对齐的目标上，而不是继续增加 weakly supervised scenes。

使用 `hardlabel100_full100_80`：

| 设置 | 值 |
|---|---:|
| WildRGBD scenes | 50 |
| DL3DV scenes | 50 |
| 训练 scenes 总数 | 100 |
| WildRGBD full-view frames | 100 |
| DL3DV full-view frames | 80 |
| 每个 scene 的 hard subsets | 12 |
| VGGT cache jobs | 1,300 |

每个 scene 的 subset methods：

- `random20_seed000` ... `random20_seed004`
- `random50_seed000` ... `random50_seed002`
- `uniform20`
- `uniform50`
- `contiguous20_seed000`
- `contiguous50_seed000`

这会生成真正的 hard native labels：

```text
full images -> VGGT-OMEGA -> full depth/pose/register cache
hard subset images -> VGGT-OMEGA -> subset depth/pose/register cache
same subset images in full cache vs subset cache -> native geometry errors
```

主 hard-label target：

```text
target_error =
    zscore_by_scene(pose_rotation_mean_deg)
  + zscore_by_scene(pointmap_rmse_norm)
  + zscore_by_scene(depth_log_rmse)
```

readout 应把较低 `target_error` 的 subset 排在较高 `target_error` 的 subset 前面。LTM30 继续作为 held-out validation，不参与训练。

## End-to-End 决策

readout 可以融入 Stage 2 selector 的训练图中，但第一版不建议 joint training。

推荐顺序：

1. parameter-free mean pooling baseline。
2. 单独训练/校准 readout。
3. 冻结 readout，训练 `0004` fixed-K selector。
4. 只有当 frozen-readout selector 的 hard validation 稳定后，才尝试 selector + readout joint fine-tuning。

不建议一开始端到端联合训练的原因：

- readout 和 selector 会 co-adapt，selector 可能学会利用 readout 漏洞，而不是选出真实更好的图像。
- readout 目标移动会让 selector loss 不稳定。
- soft relaxed top-K 与 hard VGGT subset 之间本来就有 gap，joint training 会更难定位失败原因。
- 完整穿过 VGGT-OMEGA 和 hard top-K 的端到端训练不现实；可行的“端到端”只是 cached-token selector/readout 端到端。

如果后续做 joint fine-tuning，应使用小学习率、readout distillation anchor、scene-held-out hard validation，并保留 frozen-readout checkpoint 作为回退。

## 指标

Primary validation metrics：

- readout score 与 `pose_rotation_mean_deg` 的 scene-wise mean Spearman。
- readout score 与 `pointmap_rmse_norm` 的 scene-wise mean Spearman。
- readout score 与 `depth_log_rmse` 的 scene-wise mean Spearman。
- best-score vs best-native-quality match rate。
- 跨 scenes 的 sign consistency。

Secondary metrics：

- Pearson correlation。
- sensor `gt_depth_absrel_mean` correlation。
- random/uniform/k-center candidates 中的 hard subset rank accuracy。
- scalar predicted native error 的 calibration error。

在 pose convention 审计完成前，不把 direct sensor `gt_pose_*` 作为 gate。

## Gate

This experiment passes only if a trained/calibrated readout improves over mean-pooled register cosine on scene-held-out validation.

建议的初始通过阈值：

- Native geometry target mean Spearman improves by at least `+0.10` over mean pooling on two of three primary targets.
- Expected sign is correct on at least `28/30` LTM30 validation scenes for the strongest target, or equivalent held-out set rate.
- Best-method match improves over mean pooling on at least two primary targets.
- No regression larger than `0.05` mean Spearman on sensor depth sanity metric.

如果达不到这些阈值，`0004` 应先以 mean pooling 作为 baseline objective，并把 trained readout 视为尚未准备好。

## 推荐下一步

在 selector training 前先创建 readout-calibration dataset builder：

1. Keep the existing LTM30 as held-out validation.
2. Sample additional pose/depth scenes from `data/raw/ltm_datasets`.
3. Generate 10-50 candidate subsets per scene.
4. Cache frozen VGGT-OMEGA camera/register/depth/pose outputs.
5. Compute native subset-vs-full metrics.
6. Train pooled MLP and attention readout.
7. Promote the best readout checkpoint into `0004` only after the gate passes.
