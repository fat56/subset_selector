# 0005 Image-Only Teacher/Student Selector

## 元数据

- 实验 ID: `0005_image_only_teacher_student_selector`
- 阶段: `stage2`
- 状态: design-draft
- 创建日期: 2026-06-09
- 依赖: `0003_stage2_readout_calibration`, `0004_stage2_fixed_k_selector_training`
- 核心约束: student selector 推理时不能读取 VGGT-OMEGA tokens/features。

## 背景

`0004` 证明了一个重要边界：如果 selector 的输入是 full scene 已经经过 VGGT-OMEGA 得到的 per-frame compact tokens，那么它更像离线后处理/压缩器，而不是真正的 pre-VGGT image subset selector。这个方向可以用来分析 labels 和 candidates，但不符合最终目标。

本实验重新定义目标：训练阶段允许使用 full VGGT-OMEGA 作为 teacher 生成监督信号；推理阶段 student 只能输入原始图像或 cheap image features，然后选出 top-K 子集，再把这个子集交给 VGGT-OMEGA / FastGS / 3DGS。

## 问题

能否训练一个 image-only 或 cheap-feature student selector，在不预先运行 full VGGT-OMEGA 的情况下，选出比 `uniform20` 更能代表整个 scene 三维结构的 fixed-K 图像子集？

## 核心判断

这个方向是合理的，但难点不在“让网络更大”，而在如何让轻量 student 学到 full VGGT-OMEGA 的跨图像信息交换效果。

VGGT-OMEGA 内部有多层交替注意力，可以在 full scene 的图像间交换信息。如果 student 直接复制 24 层交替 attention，参数量和计算量都会接近 teacher，缺少实践意义。因此 0005 不追求复刻 VGGT，而是学习一个低成本近似：

- 用 cheap backbone 提取每张图的 image embedding。
- 用少量 scene memory/register slots 表达当前 scene 已覆盖的信息。
- 用 marginal gain 或 candidate ranking 学习“新图像加入后，对 scene 表征补充了多少”。
- 用 teacher labels 提供监督，但推理时不调用 teacher。

## 查阅资料后的设计依据

- AsyncMDE 提出 heavy foundation model 在后台提供高质量 spatial features，lightweight foreground model 复用 cached spatial memory 并异步更新；这支持“teacher 只用于训练/刷新，student 推理时靠轻量 memory amortization”的方向。参考: https://arxiv.org/abs/2603.10438
- DINOv2 提供跨数据分布的 self-supervised visual features，适合作为第一版 frozen cheap image backbone。它不是最终轻量方案，但适合先验证 image-only 信号是否存在。参考: https://arxiv.org/abs/2304.07193
- TokenLearner 说明可以用少量 learned tokens 表达图像/视频输入，适合把每帧 patch features 压缩成 `4-8` 个 tokens，再交给 scene memory。参考: https://arxiv.org/abs/2106.11297
- Perceiver IO 使用 latent bottleneck 处理大规模输入，适合作为 scene-level memory/register slots 的结构参考，避免 full `N x N` attention。参考: https://arxiv.org/abs/2107.14795
- LSTR 用 long/short memory 做在线视频建模，可参考到 selector：长期 memory 表达 scene coverage，短期窗口处理局部视角变化。参考: https://arxiv.org/abs/2107.03377
- Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in Dynamic Environments 把 keyframe selection 和 3D reconstruction quality 直接挂钩，并使用动态阈值/误差信号优于静态间隔采样；这支持“不要只做 uniform，而要估计当前帧对重建质量的 marginal gain”。参考: https://arxiv.org/abs/2510.23928

## 对当前思路的判断

你的担心是对的：如果 selector 在推理时先跑 full VGGT-OMEGA，再根据 VGGT tokens 打分，那么它只能节省后处理或 3DGS 的成本，不能节省 VGGT 本身的成本。这个网络的实践意义会变弱。

所以 0005 的边界必须很硬：

- teacher 可以很重，可以在训练 label 生成阶段跑 full VGGT-OMEGA。
- student 必须 image-only 或 cheap-feature-only。
- student 可以用 memory/register slots 近似跨图像信息交换，但不能复刻 VGGT 的 24 层 alternating attention。
- 最重要的训练目标不是“预测某个单帧好不好”，而是“预测这张图加入当前已选集合后，对 scene coverage / teacher score 的增益有多大”。

## 与 0004 的区别

| 项目 | 0004 | 0005 |
|---|---|---|
| 推理输入 | 已经由 VGGT-OMEGA 提取的 per-frame tokens | 原始图像或 cheap image features |
| Teacher 使用 | 训练/推理都依赖 VGGT feature cache | 只在训练/label 生成时使用 |
| 目标 | 从 VGGT features 中选 subset | 在 VGGT 之前选 subset |
| 主要价值 | 离线诊断、后验压缩、验证 label 可学性 | 真正节省 VGGT/3DGS 下游成本 |
| 失败边界 | val 有信号但 test 不泛化 | 尚未验证 |

## 候选架构

### Main V1: batch cheap-feature candidate selector

这是最稳的第一版，不做在线决策，先让 student 在 full scene 的 cheap features 上选择 top-K。

输入：

- 每帧 frozen DINOv2-S/ViT-S global embedding，或 MobileNetV3/ConvNeXt-Tiny embedding。
- 可选 patch summary：mean/max/std 或 TokenLearner 压缩出的 `4-8` 个 image tokens。
- `frame_pos`，表示时间位置。
- 简单图像质量特征：blur、曝光、纹理量、动态模糊等。

网络：

```text
image_i
  -> frozen cheap backbone
  -> per-frame feature x_i
  -> projection MLP
  -> scene memory slots M=8 or 16
  -> cross-attention: memory attends to frames
  -> frame attends to memory summary
  -> score_i
  -> topK(score)
```

关键点：

- 不做 full `N x N` self-attention，避免变成小号 VGGT。
- 用 `M` 个 memory slots 做 latent bottleneck，复杂度约 `O(N*M)`。
- 第一版可以 frozen cheap backbone，只训练 projector、memory attention 和 score head。

训练目标：

- 复用 hard-native candidate labels，训练 student 给更好 candidate 更高分。
- 加 `uniform-gated` loss：默认不偏离 uniform，只有 teacher margin 足够大时才鼓励 non-uniform。
- 加 cheap-feature coverage/diversity 正则，防止只选清晰但重复的图。

### Main V2: streaming memory selector

这是更接近真实使用场景的版本：图像按顺序进入，selector 持续维护 scene memory，并估计当前图像的 marginal gain。

网络：

```text
memory_{t-1}, x_t
  -> gain_t = ScoreHead(x_t, memory_{t-1})
  -> accept/update gate
  -> memory_t = MemoryUpdate(memory_{t-1}, x_t, gain_t)
```

两种推理模式：

- 离线 topK: 所有帧都过一遍 cheap backbone 和 memory scorer，最后取 gain/score topK。
- 在线 budgeted selection: 场景流式到来时动态保留当前最有价值的 K 张。

这个版本可以参考 AsyncMDE 的思想：heavy teacher 的信息只在训练阶段塑造 memory 目标；推理时 lightweight student 用 memory 追踪场景覆盖状态。

### Main V3: marginal-gain / submodular surrogate

把 selector 问题显式写成“每张新图带来的三维表征增益”。

Teacher label 生成：

```text
S_0 = empty or uniform seed
for step in 1..K:
    choose frame j maximizing teacher_gain(S + j, full_scene)
```

其中 `teacher_gain` 可以由以下近似得到：

- subset-vs-full VGGT-native target_error 改善。
- teacher register/readout embedding 与 full embedding 的 cosine 改善。
- pose/pointmap/depth consistency 改善。
- 若完整 greedy 太贵，先在 candidate pool 上估计 pairwise preference。

Student 学习：

- `gain_i = f(x_i, memory)`，拟合 teacher greedy gain。
- pairwise loss: `gain_good > gain_bad`。
- topK loss: selected set 的 teacher score 高于 uniform/random。

这条路线最符合“场景中一张图像进来，场景总分提高最大的 top-K 为最优”的直觉。

## 第一版推荐

我建议 0005 从 Main V1 开始，Main V2/Main V3 作为后续扩展。

理由：

- Main V1 先验证 cheap image features 是否含有足够的 3D subset selection 信号。
- 它可以复用 0003/0004 已有 hardlabel300，先不新增大规模 VGGT cache。
- 如果 Main V1 连 val 都不能超过 uniform，就不必急着做 streaming memory。
- 如果 Main V1 有 val/test 正信号，再扩展到 Main V2 的 online memory。

## 数据与 label 计划

### Label v0: 复用 hardlabel300

直接复用：

- `runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv`
- candidates: `uniform20`, `random20_seed000-004`, `contiguous20_seed000`

用途：

- 验证 image-only student 是否能从 cheap features 学到 hard-native candidate ranking。
- 作为 smoke/first run，不作为最终结论。

### Label v1: hardlabel1000 + richer candidates

如果 v0 有信号，扩到至少 `1000` scenes：

- DL3DV、WildRGBD、ScanNet、BridgeData、NYUv2 尽量均衡。
- val/test 各不少于 `100` scenes。
- 每个 scene 至少 `15-25` 个 candidates。

候选族：

- `uniform20`
- `random20_seed000-009`
- `uniform_jitter20_seed000-004`
- `dino_kcenter20`
- `cheap_feature_kcenter20`
- `motion_spread20`
- `contiguous20_seed000`

## 训练目标

第一版 loss:

```text
L = L_pairwise_candidate_rank
  + 0.3 * L_oracle_ce_if_margin_large
  + 0.2 * L_uniform_gate
  + 0.05 * L_coverage_regularizer
```

解释：

- `L_pairwise_candidate_rank`: 同一 scene 内 target_error 更低的 candidate 分数更高。
- `L_oracle_ce_if_margin_large`: 只有 non-uniform oracle 比 uniform 好到超过 margin 时才用 CE 强推。
- `L_uniform_gate`: 默认保持 uniform，防止学生模型过度偏离强 baseline。
- `L_coverage_regularizer`: 用 cheap feature space 约束 selected frames 覆盖 full scene。

## 评估口径

硬约束：

- student inference 不允许读取 VGGT-OMEGA tokens/features。
- 允许读取原始 RGB、frame order、cheap backbone embedding、图像质量统计。

Primary metric:

- `uniform_minus_student_error = mean(target_error_uniform20) - mean(target_error_student)`。
- 大于 `0` 才说明 student 比 uniform 更好。

Secondary metrics:

- `win_rate_vs_uniform`
- `oracle_top1_rate`
- `pairwise_accuracy`
- deviation rate
- cheap-feature inference time
- 是否能在 held-out test 保持正提升

Promotion gate:

- val 选出的 checkpoint/gate 必须在 held-out test 上 `uniform_minus_student_error > 0`。
- test deviation rate 不能靠极少数偶然样本支撑。
- 若 test 最优策略仍是 uniform fallback，则不进入 hard subset VGGT rerun。

## 风险

- Cheap features 可能主要表达语义/外观，不足以表达三维 coverage。缓解：加入 patch statistics、motion/overlap proxy、feature k-center candidate labels。
- Teacher labels 噪声大，uniform baseline 很强。缓解：使用 margin-aware gate，默认 uniform fallback。
- Streaming selector 容易受输入顺序影响。缓解：先做 offline batch selector，再做 streaming。
- 如果使用 DINOv2，推理成本仍不为零。缓解：先确认信号，再蒸馏到 MobileNetV3/ConvNeXt-Tiny。
- 只用 candidate ranking 可能限制上限。缓解：后续加入 greedy marginal-gain teacher。

## 决策

0005 值得单独开线。它更贴近原始目标，也能解释 0004 的概念边界：0004 是 VGGT-feature 后处理 selector，0005 才是 pre-VGGT image-only selector。
