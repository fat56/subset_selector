# 复盘

## 当前判断

0005 值得单独开线，但 `Main V1 hardlabel300 candidate-rank` 第一轮没有得到可推广提升。

原因是 0004 的核心输入是 full scene 的 VGGT-OMEGA compact features，推理阶段仍需要先跑 VGGT，因此不符合“先选子集，再运行 VGGT/3DGS”的原始目标。

0005 的目标更明确：用 VGGT-OMEGA teacher 生成监督，但训练一个推理时只看原始图像或 cheap image features 的 student selector。

## 已完成的 Main V1

本轮实现并运行了两个 image-only backbone：

- `ConvNeXt-Tiny`
- `DINOv2-S/ViT-S`

网络是 `memory_candidate_set`：

```text
image_i
  -> frozen cheap backbone
  -> image-only frame feature
  -> projection MLP
  -> latent memory slots
  -> candidate mask pooling
  -> candidate score
```

训练 label 复用 `0003` hardlabel300 的 hard-native `target_error`，只在已有 candidates 中做 ranking：

- `uniform20`
- `random20_seed000-004`
- `contiguous20_seed000`

student 输入没有 VGGT-OMEGA `camera_token`、`register_tokens` 或 full VGGT output。

## 结果判断

两个 run 的 best-val checkpoint 都退回 `uniform20`：

| Run | Val uniform - student | Test uniform - student | 判断 |
|---|---:|---:|---|
| `main_v1_convnext_tiny_candidate_rank` | `0.0000` | `0.0000` | 未超过 uniform |
| `main_v1_dinov2_vits14_candidate_rank` | `0.0000` | `0.0000` | 未超过 uniform |

这不是代码失败，而是一个很有用的诊断：

- candidate pool 里 `uniform20` 是 `217/300` 个 scene 的 oracle。
- 偏离 uniform 的 epoch 在 val 上都变差。
- pairwise accuracy 约 `0.69-0.72`，说明模型能读到部分排序信号，但这个信号不足以支撑可靠的 top candidate deviation。
- DINOv2-S 没有明显优于 ConvNeXt-Tiny，至少在 global embedding + hardlabel300 candidate-rank 这个设定下不是瓶颈。

## 建议路线

第一步已经回答了一个基础问题：

> cheap image features 是否足以预测 hard-native candidate ranking？

在当前 hardlabel300 candidate pool 下，答案是：不足以超过 `uniform20`。但这个结论不等价于 image-only selector 没希望，因为当前 label 设计过于偏向 uniform，且 candidate family 太窄。

下一步不建议继续只换 backbone 或堆训练轮次。更值得推进的是改 teacher label/candidate 设计：

- 增加 `uniform_jitter20`、`cheap_feature_kcenter20`、`dino_kcenter20`、`motion_spread20` 等更有竞争力的 candidates。
- 把 objective 从“候选 top1 分类”改为 margin-aware gate：只有当 teacher 明确优于 uniform 时才允许 deviation。
- 构建 `Main V3 marginal-gain teacher`，让 student 学“加入当前集合后的增益”，而不是在偏 uniform 的候选池里猜 top1。
- 如果仍用 Main V1，应该先扩到 hardlabel1000 + richer candidates，而不是在 hardlabel300 上继续加 epoch。

## Richer Candidates 更新

已按上述判断实现 richer-candidate label 流程，并完成 full300 诊断：

1. `4-scene smoke` 成功，证明新增候选与 native metric 计算链路可跑通。
2. 正式 hardlabel300 cache 曾在 `1706/2700` jobs 后被 GPU/driver 状态阻塞；GPU 恢复后已补齐到 `2700/2700`。
3. 正式 labels 规模为 `4500 = 300 scenes * (7 old + 8 richer candidates)`。

full300 oracle 分布明显改善：

| Family | Oracle scenes |
|---|---:|
| `uniform_jitter20` | 192 |
| `uniform20` | 52 |
| `convnext_kcenter20` | 19 |
| `random20` | 15 |
| `dinov2_kcenter20` | 14 |
| `motion_spread20` | 8 |

这说明第一轮失败的主要原因确实是旧 candidate pool 过窄、过度偏向 `uniform20`。在 full300 上，richer-best 比 old-best 平均低 `0.4653` target_error，并且在 `233/300` 个 scene 上优于旧候选池 best。

## Student 诊断

ready108 partial 曾给过一个小正信号：

| Run | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---:|---:|---|
| ConvNeXt-Tiny ready108 | `+0.0814` | `-0.2163` | 不稳 |
| DINOv2-S ready108 | `+0.2447` | `+0.0436` | 有 partial 正信号 |

但 full300 正式训练没有保持住：

| Run | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---:|---:|---|
| DINOv2-S richer300 | `0.0000` | `0.0000` | best-val 回到 `uniform20` |
| ConvNeXt-Tiny richer300 | `-0.0215` | `0.0000` | best-val 接近 uniform，test 回到 `uniform20` |

这不是 richer labels 失败，而是当前 `memory_candidate_set` 的决策校准失败：

- 两个 backbone 的 pairwise accuracy 都约 `0.71-0.73`，说明排序信号存在。
- 但 top-1 candidate choice 一旦偏离 `uniform20`，平均 target_error 往往变差。
- 中间 epoch 会选择很多 `uniform_jitter20`，但 val/test 不稳定，说明需要显式保守 gate。

当前最合理的判断是：

- 不再继续只换 backbone 或增加 epoch。
- 对 `memory_candidate_set` 加 `uniform fallback margin` / margin-aware gate：只有非 uniform 分数超过 `uniform20` 足够多时才允许 deviation。
- 已对已有 checkpoints 做 post-hoc gate sweep，未找到稳定可推广 margin。
- 如果 gate sweep 仍失败，再进入 Main V3 marginal-gain teacher，而不是继续堆 `memory_candidate_set` 容量。

## Uniform Gate 结果

post-hoc gate sweep 的结论偏负面：

| Run | Val-selected rule | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---|---:|---:|---|
| DINOv2-S richer300 | `last`, margin `6.5008` | `+0.0016` | `-0.1196` | 不可推广 |
| ConvNeXt-Tiny richer300 | `best_uniform_improvement`, margin `0.2` | `+0.0165` | `0.0000` | test 回 uniform |

DINOv2-S 的 test oracle scan 能找到 `+0.0174`，但对应 val 为 `-0.0432`，所以不能作为可部署 gate。这个结果说明问题不只是阈值没调好，而是 score calibration 本身没有学会“值得偏离 uniform 的条件”。

因此下一步应该把保守性放进训练目标，而不是只做推理后处理：

- 对 non-uniform oracle 只有当它比 `uniform20` 好过 teacher margin 时才加 CE。
- 对没有明显优势的 scene 强制 uniform fallback。
- 或直接改为 Main V3 marginal-gain teacher，把每次加入 frame 的增益作为监督。

已完成 uniform-gated loss 正式训练：

| Run | Margin | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---:|---:|---:|---|
| DINOv2-S gated | `0.2` | `+0.0005` | `+0.0328` | 有轻微信号 |
| DINOv2-S gated | `0.5` | `+0.0166` | `-0.6146` | 失败 |
| ConvNeXt-Tiny gated | `0.2` | `0.0000` | `0.0000` | 回到 uniform |

DINOv2-S `margin=0.2` 是 Main V1 目前最好的结果，但 val 只比 uniform 好 `+0.0005`，而且 val-selected post-hoc gate 的 test 变成 `-0.0105`。所以它可以作为一个“方法方向有信号”的 checkpoint，不应作为最终 selector 推进。

当前判断：

- `uniform-gated` 比纯 post-hoc gate 更合理，能让 test 出现 `+0.0328` 的小正收益。
- 但 Main V1 的整组候选 top-1 分类仍然 calibration 不稳。
- 下一支应切到 Main V3 `marginal-gain teacher`：训练 student 预测“把某帧加入当前 subset 后，对 teacher score 的边际增益”，再用 greedy/beam 选 topK。

## Frame-Score 对照

已跑 `memory_frame_score` 对照：

| Run | Val `uniform - learned` | Test `uniform - learned` | Val pairwise | Test pairwise | 判断 |
|---|---:|---:|---:|---:|---|
| Candidate-set DINOv2 gated m=0.2 | `+0.0005` | `+0.0328` | `0.7281` | `0.7249` | 当前最好 |
| Frame-score DINOv2 gated m=0.2 | `0.0000` | `0.0000` | `0.6951` | `0.7275` | 回到 uniform |

frame-score 的候选 score 是 selected frame scores 的平均值，它没有学出稳定 deviation。这个对照说明问题不是“把输出粒度从 candidate 改成 frame 就会自然变好”，而是需要显式表示 coverage/diversity 或 marginal gain。

下一版 Main V3 需要更接近子模优化：

- 用已有 richer candidate cache 构造 greedy/beam teacher。
- 监督每一步 `S -> S + i` 的增益，而不是只监督最终候选 top1。
- 若离线近似有正信号，再补跑更多 candidate cache 做 true teacher。

## Ridge-Gain 近似

已实现 `frame_target_mode=ridge_gain`：

- 以 `uniform20` 为基准，定义 `utility = uniform20_error - candidate_error`。
- 用 candidate masks 做 ridge regression，把 candidate utility 近似分摊到 frames。
- 训练 `memory_frame_score` 同时做 candidate ranking、uniform-gated CE 和 frame target regression。

结果：

| Run | Weight | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---:|---:|---:|---|
| Ridge-gain frame-score | `0.20` | `+0.0148` | `0.0000` | test 回 uniform |
| Ridge-gain frame-score | `0.05` | `+0.0130` | `0.0000` | test 回 uniform |

target 分布显示这个 proxy 较噪：`mean=-1.0240`，`std=2.4872`，`positive_fraction=0.3351`，并且触达 `[-5, +5]` clip。它能制造一点 val deviation，但不能超过 candidate-set gated 的 test `+0.0328`。

结论：不能继续依赖现有候选的线性反解来代表 marginal gain；下一步如果还要推进 Main V3，应补跑小规模 true step-gain cache，而不是只调 ridge 权重。

## True Swap-Gain 复盘

为了更接近真实 marginal-gain，我们补跑了围绕 `uniform20` 的 VGGT-native swap candidates：

- `swapgain20_dino1_rank000/001`: 用 DINOv2-S feature 找到与 uniform base 差异大的候选帧，做单帧替换。
- `swapgain20_dino2_seed000/001`: 做双帧替换。
- 每个 scene 新增 `4` 个 subset，300 scene 共新增 `1200` 个 VGGT cache jobs。

300-scene label 诊断是正面的：

| 指标 | 数值 |
|---|---:|
| `swap_best_win_rate_vs_uniform` | `0.7300` |
| `swap_oracle_rate` | `0.2433` |
| `uniform_minus_best_swap_mean` | `+0.3637` |
| `swapgain20` oracle scenes | `73 / 300` |

这说明“局部替换 uniform subset”确实能造出比 `uniform20` 更好的候选，teacher 不是空信号。

但 Main V4 student 仍没有泛化：

| Run | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---:|---:|---|
| `swapgain300_gated_m02` | `+0.0095` | `-0.3055` | 过度偏离 |
| `swapgain300_gated_m05` | `-0.0020` | `-0.0087` | 近似 uniform，但略负 |
| `swapgain300_gated_m10` | `+0.0359` | `-0.0910` | val/test 反转 |
| `swapgain300_gated_m05_adv02` | `+0.0037` | `-0.5211` | advantage regression 失败 |

post-hoc gate scan 也没有找到可部署规则：val-selected gate 在 test 上均为负；test-oracle scan 最多回到 `uniform20`，即 test `0.0000`。

当前判断：

- 0005 的 teacher label 方向是有信号的，尤其是 true swap-gain。
- 失败点在 student calibration：模型知道一些 candidate ranking，但不知道何时该相信 non-uniform。
- 简单调 `uniform_gate_margin` 或加入 `uniform_advantage_loss` 不足以解决这个问题。
- 下一步应把问题拆成两头：先训练 `advantage/gate head` 判断“是否值得偏离 uniform”，再在 gate 通过时选择候选；或者直接构建 step-level marginal-gain teacher。

## Explicit Gate Head 复盘

已完成第一版 binary advantage/gate head。它回答了一个更具体的问题：

> 如果把“是否偏离 `uniform20`”单独建模，能不能解决 Main V4 的 calibration 失败？

答案目前是否定的。显式 gate head 在 val 上能学出明显正收益，但 test 上仍然失败：

| Run | Val-selected rule | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---|---:|---:|---|
| `gate_head_aw1_gw05` | `gate_logit >= 1.0` | `+0.1120` | `-0.1174` | val/test 反转 |
| `gate_head_pm05_aw05_gw1` | `gate_logit >= -1.0` | `+0.1303` | `-0.3096` | gate 更激进，test 更差 |

这说明 Main V4 的失败不是“没有单独 gate head”这么简单。当前 300-scene split 上，student 可以拟合 val 里哪些 non-uniform 候选值得尝试，但这个规则到 held-out test 后会选错，尤其倾向选择不稳定的 `uniform_jitter20`。

当前判断：

- true swap-gain teacher 是有效的，label 方向不应被放弃。
- global DINOv2-S embedding + scene-level memory 对三维 coverage 的表达可能不足，尤其缺少局部视角变化、可见性和运动跨度信息。
- 300 scene 的 train/val/test 可能太小，val-selected gate 容易偶然对齐。
- 下一步不建议继续只调 `advantage_weight` / `gate_weight`。更高价值的方向是扩大到更多 scene、做 dataset-balanced split robustness，或将 student 输入升级到 patch-summary / temporal motion proxy，再重新训练 gate。

### Seed Robustness 判断

补跑 `aw1_gw05` 的 `5` 个 seed 后，结论更清楚：

- Val-selected gate 在 `5/5` 个 seed 上都是正收益，val 均值 `+0.1084`。
- Held-out test 只有 `1/5` 个 seed 为正，test 均值 `-0.1172`。
- 唯一正的 seed `20260610` 在 test 上大量选择 `swapgain20`，达到 `+0.1163`；但另一个同样激进的 seed `20260611` 选择很多 `uniform_jitter20`，test 掉到 `-0.3548`。

因此现在不能把 `20260610` 当成可推广成果。它更像一个有价值的提示：只要 split 中 val 能正确约束 gate，swap-gain candidate 确实可能带来正收益；但目前 global image embedding 没有稳定地区分“该选 swapgain”与“会被 jitter 误导”的场景。

下一步选择应更偏工程诊断：

- 做 larger/balanced split，降低 30-scene val 的偶然性。
- 引入 patch-summary / motion-aware features，避免只靠 global DINOv2-S embedding。
- 保留 explicit gate head 架构，但不要再只在 300-scene、同一 feature 上调 loss 权重。

## 当前风险

- `uniform20` 是很强 baseline，本轮已确认 student 偏离后 val 变差。
- hardlabel300 样本太少，仍可能出现 0004 的 val/test 反转。
- cheap image backbone 的语义特征未必等价于三维 coverage。
- global embedding 可能丢失视角覆盖信息，后续可能需要 patch summary / TokenLearner tokens / optical-flow-like motion proxy。

## 下一步

- 不把本轮 checkpoint 推进到 hard subset VGGT rerun。
- 保留 ConvNeXt-Tiny 和 DINOv2-S feature cache 作为后续 ablation 资产。
- 保留 DINOv2-S gated `margin=0.2` 作为当前最好 image-only checkpoint。
- 保留 true swap-gain labels/cache 作为下一版 gate/head 训练资产。
- 若继续 0005，应优先做 larger/balanced split + patch/motion-aware gate，或 step-level marginal-gain teacher，而不是继续调同一个 candidate-set top1 loss。
