# 结果

## 当前状态

0005 已推进到 `Main V5 explicit gate head`。当前主结论是：

- true swap-gain teacher label 有明确正信号，说明“围绕 `uniform20` 做局部替换”确实能产生更好的 fixed-K subset。
- image-only student 仍然没有稳定超过 `uniform20`。显式 gate head 能在 val 上学出偏离 uniform 的规则，但 held-out test 变差。
- 因此当前不推进 checkpoint 到下游 hard subset VGGT rerun；下一步应优先扩大/重切 split、引入更强的 geometry-aware image tokens，或构建 step-level marginal-gain teacher。

历史上 `Main V1` 已完成第一轮 image-only hardlabel300 candidate-rank 实验：

- `main_v1_convnext_tiny_candidate_rank`
- `main_v1_dinov2_vits14_candidate_rank`

这两个 run 都严格遵守 0005 边界：student 输入是原始图像经过 cheap backbone 得到的 image-only features，训练/推理均不读取 VGGT-OMEGA tokens/features。

## 指标

| Run | Student input | Teacher labels | Val uniform - student | Test uniform - student | 结论 |
|---|---|---|---:|---:|---|
| `main_v1_convnext_tiny_candidate_rank` | ConvNeXt-Tiny global embedding + image stats | hardlabel300 | `0.0000` | `0.0000` | best-val checkpoint 退回 `uniform20` |
| `main_v1_dinov2_vits14_candidate_rank` | DINOv2-S/ViT-S global embedding + image stats | hardlabel300 | `0.0000` | `0.0000` | best-val checkpoint 退回 `uniform20` |

解释：

- `uniform - student > 0` 才表示 student 比 `uniform20` 更好。
- 两个 backbone 的 best-val checkpoint 都没有得到正提升。
- 训练中存在偏离 `uniform20` 的 epoch，但 val `uniform - student` 都为负，说明偏离 uniform 会变差。

## 详细结果

### `main_v1_convnext_tiny_candidate_rank`

- Feature cache: `caches/image_features/0005/hardlabel300_convnext_tiny`
- Run dir: `runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_candidate_rank`
- Feature cache 耗时: `125.5s`
- Training 耗时: `26.17s`
- Feature cache 大小: `44.3 MiB`
- Backbone feature dim: `768`

Best-val checkpoint:

| Split | Learned mean error | Uniform20 mean error | Uniform - learned | Oracle top1 | Pairwise acc |
|---|---:|---:|---:|---:|---:|
| train | `-0.8582` | `-0.8582` | `0.0000` | `0.7167` | `0.7074` |
| val | `-1.0574` | `-1.0574` | `0.0000` | `0.7667` | `0.7186` |
| test | `-0.9917` | `-0.9917` | `0.0000` | `0.7333` | `0.6994` |

### `main_v1_dinov2_vits14_candidate_rank`

- Feature cache: `caches/image_features/0005/hardlabel300_dinov2_vits14`
- Run dir: `runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_candidate_rank`
- Feature cache 耗时: `141.8s`
- Training 耗时: `23.25s`
- Feature cache 大小: `24.6 MiB`
- Backbone feature dim: `384`

Best-val checkpoint:

| Split | Learned mean error | Uniform20 mean error | Uniform - learned | Oracle top1 | Pairwise acc |
|---|---:|---:|---:|---:|---:|
| train | `-0.8582` | `-0.8582` | `0.0000` | `0.7167` | `0.6992` |
| val | `-1.0574` | `-1.0574` | `0.0000` | `0.7667` | `0.6884` |
| test | `-0.9917` | `-0.9917` | `0.0000` | `0.7333` | `0.7251` |

## Dataset 诊断

本轮 hardlabel300 candidate pool:

- scenes: `300`
- split: train `240`, val `30`, test `30`
- datasets: DL3DV `150`, WildRGBD `150`
- candidates per scene: `uniform20`, `random20_seed000-004`, `contiguous20_seed000`
- oracle counts: `uniform20 = 217`, `random20 = 82`, `contiguous20 = 1`

这说明当前 labeled candidates 中 `uniform20` 本身非常强，占 oracle 的 `72.3%`。因此 Main V1 candidate-rank 模型最稳的 validation 策略就是选择 `uniform20`，而不是学习非 uniform deviation。

## Richer Candidates 进度

为解决 hardlabel300 candidate pool 过于偏向 `uniform20` 的问题，已实现 richer-candidate label 生成流程，并完成 4-scene smoke。

新增候选：

- `uniform_jitter20_seed000-004`
- `convnext_kcenter20_seed000`
- `dinov2_kcenter20_seed000`
- `motion_spread20_seed000`

4-scene smoke:

- Run dir: `runs/0005_image_only_teacher_student_selector/richer_candidates_smoke4`
- Cache root: `caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_smoke4_images512`
- Jobs: `36 = 4 scenes * (1 full + 8 subset)`
- Result: cache + labels 成功。
- Smoke merged labels: `2132` rows。
- Smoke oracle 新增 `uniform_jitter20 = 4`，说明 richer candidates 确实能制造比旧候选更有信息量的监督。

正式 hardlabel300 richer-candidate cache:

- Run dir: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300`
- Cache root: `caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_images512`
- Total jobs: `2700 = 300 scenes * (1 full + 8 subset)`
- 当前完成: `2700 / 2700`
- 当前缺失: `0 / 2700`
- 完整 scene: `300 / 300`
- 当前 cache 占用: 约 `167G`

历史阻塞：

- 正式 cache 过程中出现 `CUDA error: unspecified launch failure`。
- 随后 `nvidia-smi` 只枚举到一张 RTX 5090，单卡 resume 又在 CUDA 初始化阶段报 `CUDA unknown error`。
- GPU/driver 恢复后已用 `selector0005_richer_resume` 补齐剩余 `994` jobs，两个 worker 均 `failed=0`。

正式 merged labels:

- Labels: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_hardlabel_train_labels.csv`
- Jobs: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_cache_jobs.json`
- Rows: `4500 = 300 scenes * (7 old candidates + 8 richer candidates)`
- 数据分布: `DL3DV-ALL-480P = 150`, `wildrgbd_harrison = 150`
- 诊断输出: `full300_richer_diagnostic_summary.json`

Full300 oracle family:

| Family | Oracle scenes |
|---|---:|
| `uniform_jitter20` | 192 |
| `uniform20` | 52 |
| `convnext_kcenter20` | 19 |
| `random20` | 15 |
| `dinov2_kcenter20` | 14 |
| `motion_spread20` | 8 |

关键判断：

- `uniform20` oracle 从旧池的 `217/300` 降为 `52/300`。
- richer-best 比 old-best 平均低 `0.4653` target_error。
- richer candidates 在 `233/300` 个 scene 上优于旧候选池 best。
- `uniform_minus_oracle_error = 0.8504`，说明 full300 richer labels 确实提供了更宽的 teacher margin。

## Ready108 Partial 诊断

在 GPU 恢复前，使用 `--ready-only` 对 108 个已完整缓存的 scene 先做 partial label 诊断：

- Run dir: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_ready108`
- Labels: `1620 = 108 scenes * (7 old candidates + 8 richer candidates)`
- 数据分布: `DL3DV-ALL-480P = 57`, `wildrgbd_harrison = 51`
- 诊断输出: `partial_richer108_diagnostic_summary.json`

Oracle family:

| Family | Oracle scenes |
|---|---:|
| `uniform_jitter20` | 76 |
| `uniform20` | 14 |
| `random20` | 8 |
| `dinov2_kcenter20` | 5 |
| `convnext_kcenter20` | 5 |

关键判断：

- `uniform20` 不再垄断 oracle，从旧池的 `217/300` 强势状态降为 ready108 的 `14/108`。
- richer-best 比 old-best 平均低 `0.4439` target_error。
- richer candidates 在 `86/108` 个 scene 上优于旧候选池 best。
- `uniform_minus_oracle_error = 0.9060`，说明这批 richer labels 有足够 teacher margin。

## Ready108 CPU 训练诊断

由于当前 CUDA 不可用，先用 CPU 跑小型 `memory_candidate_set` 诊断，只作为信号检查，不作为 promotion 结论。

| Run | Backbone | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---|---:|---:|---|
| `main_v1_convnext_tiny_richer108_cpu` | ConvNeXt-Tiny | `+0.0814` | `-0.2163` | val 有信号，但 test 不稳 |
| `main_v1_dinov2_vits14_richer108_cpu` | DINOv2-S/ViT-S | `+0.2447` | `+0.0436` | partial 正信号，值得补全 300-scene |

DINOv2-S ready108 best checkpoint:

- Train: `uniform_minus_learned_error = +0.5351`
- Val: `+0.2447`, `win_rate_vs_uniform = 0.2727`, `pairwise_accuracy = 0.7076`
- Test: `+0.0436`, `win_rate_vs_uniform = 0.2000`, `pairwise_accuracy = 0.7483`
- Test learned methods: `uniform20 = 5`, `uniform_jitter20 = 5`

这个结果说明：一旦 candidate pool 不再过度偏向 `uniform20`，image-only student 至少可以在 partial set 上学到一点正向 deviation。正式结论仍需等 GPU 恢复后补全 `300` scene richer cache，再做双卡正式训练。

## Richer300 正式训练

完成两个 image-only backbone 的正式 richer300 双卡训练：

| Run | Backbone | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---|---:|---:|---|
| `main_v1_dinov2_vits14_richer300` | DINOv2-S/ViT-S | `0.0000` | `0.0000` | best-val 退回 `uniform20` |
| `main_v1_convnext_tiny_richer300` | ConvNeXt-Tiny | `-0.0215` | `0.0000` | best-val 接近 uniform，test 退回 `uniform20` |

DINOv2-S richer300:

- Train/Val/Test learned methods at best checkpoint 均为 `uniform20`。
- Val pairwise accuracy: `0.7111`
- Test pairwise accuracy: `0.7205`
- Val oracle: `uniform_jitter20 = 15`, `uniform20 = 9`, 其余 `6`
- Test oracle: `uniform_jitter20 = 19`, `uniform20 = 10`, `convnext_kcenter20 = 1`

ConvNeXt-Tiny richer300:

- Val learned methods: `uniform20 = 27`, `uniform_jitter20 = 2`, `motion_spread20 = 1`
- Test learned methods: `uniform20 = 30`
- Val pairwise accuracy: `0.7214`
- Test pairwise accuracy: `0.7346`

结论：

- richer labels 解决了候选池过窄的问题，但 `memory_candidate_set` 当前选择策略仍没有可靠超过 `uniform20`。
- 模型能学到 pairwise ranking 信号，但 top-1 deviation 风险很高；训练中间 epoch 会大量偏向 `uniform_jitter20`，val error 反而变差。
- 下一步不应继续只换 backbone，而应加入 `uniform fallback margin` 或 margin-aware gate：只有当 student 对非 uniform 候选有足够信心时才偏离 `uniform20`。

## Uniform Fallback Gate Sweep

已实现 gate sweep:

- Script: `scripts/evaluate_stage2_image_only_selector_gate.py`
- DINOv2-S output: `runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_richer300/uniform_gate_scan.json`
- ConvNeXt-Tiny output: `runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_richer300/uniform_gate_scan.json`

方法：

- 对 `best_uniform_improvement.pt`、`best_val_error.pt`、`last.pt` 分别评估。
- 默认选择 `uniform20`。
- 若最佳 non-uniform score 超过 `uniform20` score 至少 `margin`，才允许 deviation。
- 在 val 上选 margin，再看 test。

结果：

| Run | Val-selected checkpoint / margin | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---|---:|---:|---|
| DINOv2-S richer300 | `last`, margin `6.5008` | `+0.0016` | `-0.1196` | val 微弱正，test 失败 |
| ConvNeXt-Tiny richer300 | `best_uniform_improvement`, margin `0.2` | `+0.0165` | `0.0000` | val 微弱正，test 回 uniform |

Oracle-scan 只能找到不可靠信号：

- DINOv2-S 若直接按 test oracle scan，可得到 `+0.0174`，但对应 val 是 `-0.0432`，不能作为可推广策略。
- ConvNeXt-Tiny 的 test oracle scan 仍为 `0.0000`。

结论：

- 简单 post-hoc `uniform fallback margin` 不能稳定解决 calibration。
- 当前分数尺度没有把“何时值得偏离 uniform”学出来。
- 下一步需要在训练目标里显式加入 `uniform-gated` / margin-aware loss，而不只是推理时扫阈值。

## Uniform-Gated Loss

已在 `scripts/run_stage2_image_only_selector_training.py` 加入 `--uniform-gate-margin`：

- 若 `oracle_error + margin < uniform20_error`，CE target 使用 oracle candidate。
- 否则 CE target 回到 `uniform20`。
- pairwise rank loss 仍保留所有明确优劣关系。

正式跑了三组 richer300：

| Run | Gate margin | Raw Val `uniform - learned` | Raw Test `uniform - learned` | Val-selected gate Test | 判断 |
|---|---:|---:|---:|---:|---|
| `main_v1_dinov2_vits14_richer300_gated_m02` | `0.2` | `+0.0005` | `+0.0328` | `-0.0105` | 有轻微信号，但 gate sweep 不稳 |
| `main_v1_dinov2_vits14_richer300_gated_m05` | `0.5` | `+0.0166` | `-0.6146` | `-0.3997` | 过度偏离，失败 |
| `main_v1_convnext_tiny_richer300_gated_m02` | `0.2` | `0.0000` | `0.0000` | `0.0000` | 完全退回 uniform |

DINOv2-S `margin=0.2` 是目前 Main V1 最好的 checkpoint：

- Val learned methods: `uniform20 = 21`, `uniform_jitter20 = 9`
- Test learned methods: `uniform20 = 24`, `uniform_jitter20 = 6`
- Test `uniform - learned = +0.0328`
- Test oracle top1 rate: `0.4000`

但这个提升还不够稳：

- Val 正幅度只有 `+0.0005`，接近噪声。
- post-hoc gate 在 val 上选到更高 `+0.0394` 后，test 变成 `-0.0105`。
- `margin=0.5` 和 ConvNeXt 对照都没有保持正收益。

结论：

- 把保守性写入训练目标确实比单纯 post-hoc gate 更有希望。
- 但 Main V1 的 candidate-set classifier 仍没有稳定学会“何时偏离 uniform”。
- 0005 下一步应转向 Main V3 `marginal-gain teacher`，让 student 学每帧对当前集合的增益，而不是直接在整组候选里做 top-1 分类。

## Frame-Score Baseline

为了验证“逐帧打分”是否更稳，复用了训练脚本里的 `memory_frame_score` 模式：

- Run: `main_v3_dinov2_vits14_frame_score_gated_m02`
- Backbone: DINOv2-S/ViT-S image-only feature
- Loss: `uniform-gated`, margin `0.2`
- 模型输出每帧 score，候选集合 score 是 selected frames 的平均 score。

结果：

| Run | Model kind | Val `uniform - learned` | Test `uniform - learned` | Val pairwise | Test pairwise | 判断 |
|---|---|---:|---:|---:|---:|---|
| `main_v1_dinov2_vits14_richer300_gated_m02` | `memory_candidate_set` | `+0.0005` | `+0.0328` | `0.7281` | `0.7249` | 当前最好 |
| `main_v3_dinov2_vits14_frame_score_gated_m02` | `memory_frame_score` | `0.0000` | `0.0000` | `0.6951` | `0.7275` | 回到 uniform |

frame-score 的 val-selected post-hoc gate:

- Val `uniform - learned = +0.0130`
- Test `uniform - learned = 0.0000`

结论：

- 单纯 per-frame average scorer 没有超过 candidate-set scorer。
- 这支持一个判断：selector 需要 coverage/diversity-aware set reasoning，不能只给每帧独立质量分。
- 下一版 Main V3 应构建真正的 marginal-gain teacher：用已有 richer candidate cache 先做 greedy/beam 近似，再决定是否补跑更多 VGGT candidate cache。

## Ridge-Gain Frame Target

为了进一步逼近 marginal-gain，新增 `frame_target_mode=ridge_gain`：

- 在同一 scene 内，以 `uniform20` 为基准，把每个 candidate 的 `utility = uniform20_error - candidate_error` 作为监督。
- 用 candidate masks 解一个 ridge additive frame target，把 candidate-level utility 近似分摊到被选中的 frames。
- `memory_frame_score` 继续保留 candidate ranking / uniform-gated loss，并额外回归 frame target。

target 诊断：

- `count = 27000`
- `mean = -1.0240`
- `std = 2.4872`
- `positive_fraction = 0.3351`
- `negative_fraction = 0.5828`
- `min/max = -5.0 / +5.0`

结果：

| Run | Frame target weight | Val `uniform - learned` | Test `uniform - learned` | Val pairwise | Test pairwise | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `main_v3_dinov2_vits14_frame_score_ridge_gain_w02` | `0.20` | `+0.0148` | `0.0000` | `0.7194` | `0.7253` | val 小正，test 回 uniform |
| `main_v3_dinov2_vits14_frame_score_ridge_gain_w005` | `0.05` | `+0.0130` | `0.0000` | `0.6913` | `0.7285` | val 小正，test 回 uniform |

结论：

- ridge-gain proxy 比 plain frame-score 多一点 val 信号，但没有泛化到 test。
- 现有 richer candidate labels 不足以稳定反推出 per-frame marginal gain。
- 当前最好结果仍是 `memory_candidate_set + uniform-gated margin=0.2`，test `+0.0328`。
- 真正的 Main V3 需要补跑 `S + i` 或 small beam candidate cache，得到显式 step-level teacher gain。

## True Swap-Gain Candidates

为验证“围绕 `uniform20` 做局部替换是否能提供更接近 marginal gain 的 teacher”，新增 `scripts/run_stage2_image_only_swap_gain_labels.py`。

方法：

- 以每个 scene 的 `uniform20` 为 base subset。
- 用 DINOv2-S image-only feature 在 cheap feature space 中寻找和 uniform base 差异大的候选帧。
- 构造两类 fixed-K swap candidate：单帧替换 `swapgain20_dino1_rank000/001`，双帧替换 `swapgain20_dino2_seed000/001`。
- 对这些 subset 跑 VGGT-native metric，合并到 richer300 labels；student 输入仍然只用 image-only DINOv2 features。

300-scene 打标结果：

- Run dir: `runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300`
- Cache root: `caches/vggt_omega/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300_images512`
- 新增 VGGT cache jobs: `1200 = 300 scenes * 4 swap candidates`
- 合并后 labels: `5700 = 300 scenes * 19 candidates`
- 新增 cache size: 约 `52G`

Oracle family:

| Family | Oracle scenes |
|---|---:|
| `uniform_jitter20` | `155` |
| `swapgain20` | `73` |
| `uniform20` | `29` |
| `convnext_kcenter20` | `14` |
| `random20` | `13` |
| `dinov2_kcenter20` | `10` |
| `motion_spread20` | `6` |

关键诊断：

- `swap_best_win_rate_vs_uniform = 0.7300`
- `swap_oracle_rate = 0.2433`
- `uniform_minus_best_swap_mean = +0.3637`

结论：true swap-gain candidate 本身有明确 teacher 信号；问题已经不再是候选池完全没有 non-uniform 正例，而是 student 是否能稳定判断哪些 scene 值得偏离 `uniform20`。

## Main V4: Swap-Gain Student

在 300-scene true swap-gain labels 上完成 DINOv2-S `memory_candidate_set` 训练和保守 gate 对照：

| Run | 设置 | Raw Val `uniform - learned` | Raw Test `uniform - learned` | Val-selected gate Test | 判断 |
|---|---|---:|---:|---:|---|
| `main_v4_dinov2_swapgain300_candidate_set_gated_m02` | `uniform_gate_margin=0.2` | `+0.0095` | `-0.3055` | `-0.1680` | 过度偏离，test 失败 |
| `main_v4_dinov2_swapgain300_candidate_set_gated_m05` | `uniform_gate_margin=0.5` | `-0.0020` | `-0.0087` | `-0.0151` | 基本退回 uniform，但仍略负 |
| `main_v4_dinov2_swapgain300_candidate_set_gated_m10` | `uniform_gate_margin=1.0` | `+0.0359` | `-0.0910` | `-0.0910` | val 小正，test 失败 |
| `main_v4_dinov2_swapgain300_candidate_set_gated_m05_adv02` | `margin=0.5`, `uniform_advantage_weight=0.2` | `+0.0037` | `-0.5211` | `-0.2149` | advantage calibration 失败 |

补充：

- 所有 run 的 test-oracle gate scan 最多只能回到 `uniform20`，即 test `uniform - learned = 0.0000`。
- `m=0.2` raw best 在 val 上选 `uniform_jitter20=19`, `swapgain20=3`, `uniform20=7`；test 上仍偏向 `uniform_jitter20=16`，导致平均 worse than uniform。
- `m=0.5` 只在 val/test 各偏离 `uniform20` 一次，已经非常保守，但仍没有正收益。
- `uniform_advantage_loss` 让 score margin 拟合 `uniform_error - candidate_error`，但没有改善 test，说明简单 score-scale regression 不足以解决 calibration。

当前结论：

- true swap-gain labels 证明了局部替换 candidate 有价值。
- 当前 `memory_candidate_set` student 能学到 ranking 信号，但不能可靠学习“何时偏离 `uniform20`”。
- 继续调 `uniform-gate-margin`、post-hoc margin 或简单 advantage regression 的边际收益已经很低。
- 下一步应转向显式 binary advantage/gate head 或真正 step-level marginal-gain teacher，而不是继续在同一 candidate-set top1 目标上加 loss。

## Main V5: Explicit Gate Head

为直接验证“先判断是否值得偏离 `uniform20`，再选择 non-uniform candidate”是否能解决 calibration，新增 `scripts/run_stage2_image_only_gate_head_training.py`。

方法：

- Backbone 仍使用 DINOv2-S/ViT-S image-only feature cache，不读取任何 VGGT-OMEGA token。
- 网络复用 `memory_candidate_set` 的 latent memory contextualizer。
- 对每个 candidate 输出两个量：`advantage = uniform20_error - candidate_error` 的回归值，以及 binary `gate_logit`。
- 推理时默认选择 `uniform20`；只有最佳 non-uniform candidate 的 `advantage` 或 `gate_logit` 超过阈值才允许 deviation。
- 阈值只在 val 上选择，然后固定评估 test。

完成两个 full300 swap-gain 对照：

| Run | Loss 设置 | Val-selected rule | Val `uniform - learned` | Test `uniform - learned` | Test deviation | 判断 |
|---|---|---|---:|---:|---:|---|
| `main_v5_dinov2_swapgain300_gate_head_aw1_gw05` | `advantage_weight=1.0`, `gate_weight=0.5`, `positive_margin=0.2` | `gate_logit >= 1.0` | `+0.1120` | `-0.1174` | `0.1333` | val 有明显正信号，test 失败 |
| `main_v5_dinov2_swapgain300_gate_head_pm05_aw05_gw1` | `advantage_weight=0.5`, `gate_weight=1.0`, `positive_margin=0.5` | `gate_logit >= -1.0` | `+0.1303` | `-0.3096` | `0.4000` | 更强 gate loss 过度偏离，test 更差 |

补充诊断：

- `aw1_gw05` 的 val learned methods 是 `uniform20=23`, `uniform_jitter20=6`, `swapgain20=1`；test 是 `uniform20=26`, `uniform_jitter20=4`。
- `pm05_aw05_gw1` 的 val learned methods 是 `uniform20=18`, `uniform_jitter20=10`, `motion_spread20=1`, `swapgain20=1`；test 是 `uniform20=18`, `uniform_jitter20=11`, `motion_spread20=1`。
- 两个 run 的 test-oracle scan 最多只能回到 `uniform20`，即 test `uniform - learned = 0.0000`。

结论：

- 显式 gate head 确实比 Main V4 更会拟合 val：val 从 Main V4 最好 `+0.0359` 提高到 `+0.1303`。
- 但这个提升没有泛化，test 仍然是负数；问题更像 split/domain calibration，而不是 gate head 表达力不足。
- 继续在 300-scene fixed split 上调 gate loss 权重，风险很高；下一步更应该做 split robustness / larger scene set，或让 student 输入从 global embedding 升级到 patch-summary / motion-aware tokens。

## 记录口径

只有当 student 推理时不读取 VGGT-OMEGA tokens/features，结果才计入 0005。

如果某个 run 使用 VGGT-OMEGA per-frame tokens 作为输入，应归入 0004 或另记为 teacher/diagnostic ablation，不能作为 image-only selector 结果。
