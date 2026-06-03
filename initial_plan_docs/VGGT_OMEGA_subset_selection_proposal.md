# VGGT-OMEGA Register Token 驱动的数据子集选择研究方案

日期：2026-06-03

## 1. 出发点

本方案的核心问题是：给定一个离线图像数据集，能否自动挑选出一个尽可能小的图像子集，使这个子集仍然能够保留完整数据集所包含的全局 3D 几何理解能力，并支持后续三维重建或 VLA 等下游任务。

这个问题的动机来自两点：

1. VGGT-OMEGA 的 register tokens 具备聚合场景级信息的能力。VGGT-OMEGA 在架构中显式引入 camera token 和 scene/register tokens，并通过 register attention 让跨帧信息在部分层中主要通过 registers 交换。论文也展示了这些 registers 可以用于 VLA 和语言对齐，说明它们不只是辅助 token，而是携带了场景级、几何相关甚至语义相关的信息。

2. 大规模三维重建或 VLA 数据集中往往存在冗余。对于一个场景或一段视频，并非所有图像都同等重要。部分图像可能视角重复、纹理信息不足、运动模糊、遮挡严重，或者对全局几何覆盖贡献很小。因此，有可能通过学习或优化的方法选出一个较小但信息量高的子集。

目标可以表述为：

> 训练一个网络或选择策略，输入一个离线图像集合，输出一个最小或近似最小的图像子集。该子集经过冻结的 VGGT-OMEGA 后得到的 register 表征，应尽量接近完整图像集合经过 VGGT-OMEGA 后得到的 register 表征。

后续验证目标是：用所选子集进行 3DGS 或其他重建方法训练时，PSNR、SSIM、LPIPS、几何一致性、相机估计质量等指标相对于完整数据集或强基线仍然保持较好水平。

## 2. 总体可行性判断

这个想法整体是可行的，而且有明确的研究价值。不过需要注意，它更适合被定义为一个“预算化几何覆盖选择”问题，而不是一开始就定义成纯 RL 问题。

推荐的核心表述是：

> 学习一个 selector，使其在给定预算或稀疏约束下，选择一个图像子集，使该子集在 VGGT-OMEGA register 表征空间中最大程度保留完整集合的全局几何信息。

这里有几个关键判断：

- 使用 VGGT-OMEGA registers 作为 teacher pseudo-label 是合理的。完整集合经过 VGGT-OMEGA 后得到的 registers 可以看作一种高层几何表征。
- 只用 InfoNCE 不足以定义“最小集合”。InfoNCE 能定义“子集和完整集合是否匹配”，但不会自动让子集变小。必须加入固定预算、稀疏正则、停止策略，或者进行 Pareto 曲线搜索。
- 不建议直接逐 token 比较完整集合和子集的 registers。完整集合有 N 张图，子集有 K 张图，token 数量不同，且 token 上下文不同。更合理的是先用 readout head 将 register tokens 聚合成一个场景级 embedding，再比较两个 embedding。
- 不建议一开始使用 RL。RL 可以作为后期扩展，但样本效率低、训练不稳定、reward 延迟大。更现实的起点是 fixed-K selector、Gumbel-TopK、straight-through top-K 或 greedy imitation。
- 必须验证 register 表征距离是否和实际重建质量相关。如果 register embedding 接近，但 3DGS 的 PSNR 或几何质量没有提升，那么需要加入几何辅助 loss 或重建质量代理目标。

## 3. VGGT-OMEGA 中可利用的结构

本地代码中，`VGGTOmega.forward` 会直接输出：

```python
predictions = {
    "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
}
```

其中 `patch_token_start = 1 + num_register_tokens`。默认设置下，每帧包含：

- 1 个 camera token
- 16 个 register tokens
- 后续 patch tokens

相关代码位置：

- `vggt_omega/models/vggt_omega.py`
- `vggt_omega/models/aggregator.py`
- `vggt_omega/models/heads/text_alignment_head.py`

特别值得借鉴的是 `TextAlignmentHead`。它的做法是：

1. 取出所有帧的 camera/register tokens。
2. 加入一个可学习的 language/readout token。
3. 经过几层 self-attention。
4. 取 readout token 的输出作为整个序列的 embedding。
5. 投影并 L2 normalize。

这正好对应本方案中“用一个类似 [CLS] token 的可学习 token 读取 register 信息”的想法。因此，本方案推荐复用这种 readout 思路，但目标不是语言对齐，而是 full-set 与 subset 的几何表征蒸馏。

## 4. 问题定义

给定一个场景或序列：

```text
I = {I_1, I_2, ..., I_N}
```

希望选择一个子集：

```text
S = {I_i | i in selected_indices}, |S| = K, K << N
```

完整集合经过冻结的 VGGT-OMEGA 和 readout head 得到 teacher embedding：

```text
z_full = g(VGGT_OMEGA(I))
```

子集经过同一个冻结 VGGT-OMEGA 和同一个 readout head 得到 student/subset embedding：

```text
z_sel = g(VGGT_OMEGA(S))
```

训练目标是：

```text
z_sel 接近 z_full
```

同时：

```text
|S| 尽可能小
```

在实际训练中可以采用两种形式。

第一种是固定预算：

```text
|S| = K
```

分别训练或测试不同 K，例如：

```text
K = 5, 10, 20, 40
```

第二种是稀疏正则：

```text
L = L_feature + lambda * |S|
```

通过调节 `lambda` 得到不同大小的子集。

推荐先从固定预算开始，因为更稳定、更容易分析。

## 5. 推荐总体架构

### 5.1 Teacher 分支

Teacher 分支只用于生成伪标签，通常冻结。

```text
完整图像集合 I_all
    -> frozen VGGT-OMEGA
    -> camera/register tokens
    -> readout head g
    -> z_full
```

`z_full` 可以离线缓存，减少训练成本。

如果完整集合太大，无法一次性送入 VGGT-OMEGA，可以采用：

- 从完整集合中均匀采样一个 dense teacher set，例如 100 到 300 张图；
- 分块运行 VGGT-OMEGA，再用上层 set encoder 聚合；
- 使用层次化 teacher：先对局部窗口得到局部 register embedding，再聚合为全局 embedding。

### 5.2 Selector 分支

Selector 的输入可以有几种选择。

低成本版本：

```text
每张图 -> DINO/CLIP/VGGT image-only feature -> selector
```

更贴合目标的版本：

```text
完整集合经过 VGGT-OMEGA 后的每帧 camera/register summary -> selector
```

这后一种更强，但需要先运行完整 VGGT-OMEGA。它适用于“离线数据集压缩”，但如果目标是节省 VGGT-OMEGA 推理本身的成本，则需要换成更便宜的图像特征。

推荐的 selector 结构：

```text
per-image feature x_i
    -> MLP projection
    -> Set Transformer / Perceiver / DeepSets
    -> per-image selection score s_i
    -> top-K / Gumbel-TopK
    -> selected subset S
```

可加入的输入特征包括：

- 图像级语义特征，例如 DINO、CLIP；
- VGGT register summary；
- camera token；
- 时间戳或帧序号；
- 已知或估计相机位姿；
- 图像清晰度；
- 纹理量；
- 与其他图像的 overlap 或 matching score；
- VGGT depth confidence 的统计量。

### 5.3 Subset 分支

```text
selected subset S
    -> frozen VGGT-OMEGA
    -> camera/register tokens
    -> readout head g
    -> z_sel
```

训练时比较 `z_sel` 和 `z_full`。

需要注意：VGGT-OMEGA 中第一帧和其他帧使用的 camera/register 初始化 token 不完全一样，因此输入顺序可能影响结果。建议：

- 子集送入 VGGT-OMEGA 前按原始时间顺序排序；
- 或固定一个 anchor 规则，例如选择最高置信度图像作为第一帧；
- 在实验中加入 order ablation，检查顺序敏感性。

## 6. Readout Head 设计

推荐使用类似 VGGT-OMEGA `TextAlignmentHead` 的结构。

输入：

```text
camera/register tokens: [B, N, 17, C]
```

处理：

```text
tokens = reshape([B, N * 17, C])
readout_token = learnable parameter [1, 1, C]
tokens = concat(readout_token, tokens)
tokens = several self-attention blocks(tokens)
z = projection(tokens[:, 0])
z = L2_normalize(z)
```

这个 readout head 有两个作用：

1. 将不同数量图像的 register tokens 映射到统一维度；
2. 让模型学习“哪些 register 信息对全局几何表示最重要”。

可选设计：

- 使用 1 个 readout token，得到单一 scene embedding；
- 使用 M 个 readout tokens，得到多个 scene slots，再做 pooling；
- 用 Set Transformer 的 pooling by multihead attention；
- 用 Perceiver latent array 作为更强的聚合器。

MVP 阶段建议使用 1 个 readout token，2 到 4 层 self-attention。

## 7. Loss 设计

### 7.1 Symmetric InfoNCE

一个 batch 中有 B 个场景。对每个场景 i，有：

```text
z_full_i
z_sel_i
```

计算相似度矩阵：

```text
sim_ij = cosine(z_sel_i, z_full_j) / tau
```

对角线是正样本：

```text
i == j
```

非对角线是负样本：

```text
i != j
```

loss：

```text
L_nce = 0.5 * CE(sim, labels=diag) + 0.5 * CE(sim.T, labels=diag)
```

这就是你提到的“对角线匹配最大化，非对角线错误匹配最小化”。

优点：

- 适合 full-set 与 subset 的场景级对齐；
- 可以利用 batch 内负样本；
- 和 VGGT-OMEGA 语言对齐部分的 symmetric InfoNCE 思路一致。

风险：

- 不同场景可能很相似，会产生 false negative；
- InfoNCE 本身不控制子集大小；
- 只靠 InfoNCE 可能训练不稳定。

因此建议配合直接正样本距离。

### 7.2 Cosine 或 MSE 正样本蒸馏

```text
L_pos = 1 - cosine(z_sel_i, z_full_i)
```

或者：

```text
L_mse = ||z_sel_i - z_full_i||_2^2
```

推荐先用 cosine，因为 readout embedding 会 L2 normalize。

### 7.3 稀疏或预算 loss

如果使用 soft selection probability `p_i`：

```text
L_sparse = lambda * sum_i p_i
```

如果使用固定 top-K，则不需要这个项，但需要在不同 K 下做实验。

### 7.4 多样性与覆盖 loss

为了避免 selector 选择一堆相似图片，可以加入 diversity 或 coverage 约束。

例如，使用 facility location 风格目标：

```text
Coverage(S) = sum_j max_{i in S} sim(x_i, x_j)
```

训练时可以最大化 coverage，或把负 coverage 加入 loss：

```text
L_cov = - Coverage(S)
```

也可以加入 pose diversity：

```text
L_pose_div = 鼓励 selected cameras 覆盖更大的视角和基线
```

### 7.5 几何辅助 loss

可以把完整集合的 VGGT-OMEGA 输出作为 pseudo-label，用于监督子集输出。

可选项：

1. Depth loss

   对 selected images，比较子集运行时预测的 depth 与完整集合运行时对应图像的 depth。

   注意需要 scale normalization：

   ```text
   L_depth = | log d_sel - log d_full_aligned |
   ```

2. Pose loss

   不能直接比较绝对 extrinsics，因为可能有坐标系和尺度差异。建议比较：

   - relative pose；
   - pairwise camera distance；
   - Procrustes 对齐后的 pose；
   - rotation geodesic distance。

3. Point map loss

   用 depth 和 camera unprojection 得到 point map，再比较 scale-aligned 后的点云或局部点图。

4. Matching/token loss

   借鉴 VGGT-OMEGA 的 matching loss，用正负 token pair 做二分类或对比学习。

MVP 阶段建议先不用复杂几何 loss，只用：

```text
L = L_nce + beta * L_pos
```

等验证 register distance 和重建质量有一定相关性后，再加入 depth/pose 辅助项。

## 8. 离散选择的训练方法

子集选择是离散操作，不能直接反向传播。可选方法如下。

### 8.1 Fixed top-K with straight-through

selector 输出 score：

```text
s_i = selector(x_i)
```

前向时取 top-K：

```text
S = topK(s)
```

反向时用 straight-through estimator 或者只让 selector 通过 soft mask 学习。

优点是简单，适合 MVP。

### 8.2 Gumbel-TopK

对 score 加 Gumbel noise：

```text
y_i = s_i + g_i
```

再进行 top-K。训练早期温度较高，后期逐渐 anneal。

优点是比纯 top-K 更容易探索不同组合。

### 8.3 Soft mask relaxation

训练时不真正选图，而是学一个 soft weighted aggregation：

```text
z_soft = aggregate(p_i * feature_i)
```

然后让 `z_soft` 接近 `z_full`。之后再把 `p_i` 离散化。

优点是稳定，缺点是和真实“把子集送进 VGGT-OMEGA”的目标有差距。

### 8.4 RL

可以把选择过程建模为：

```text
state = 当前已选图像集合
action = 再选择一张图
reward = - feature_distance - lambda * subset_size + reconstruction_quality_proxy
```

但不建议作为第一版，因为：

- reward 延迟；
- 训练方差大；
- 每次 action 后重新跑 VGGT-OMEGA 或 3DGS 成本很高；
- 很难和强 greedy baseline 拉开差距。

RL 更适合后期作为 refinement。

## 9. 实验路线

### 9.1 阶段 0：强 baseline

在训练网络前，必须先建立强 baseline。

建议包括：

1. Random K
2. Uniform stride K
3. DINO/CLIP feature k-center
4. VGGT register embedding k-center
5. Facility-location greedy
6. Pose farthest point sampling
7. 图像质量过滤加 coverage
8. 类似 Skeletal Sets 的 overlap/robustness 图选择

如果 learned selector 打不过这些 baseline，说明网络或目标还不够好。

### 9.2 阶段 1：验证 register distance 与重建质量相关性

对不同 selector 和不同 K，记录：

```text
register cosine similarity
InfoNCE retrieval accuracy
subset size K/N
3DGS PSNR
3DGS SSIM
3DGS LPIPS
pose ATE/RPE
depth error
reconstruction completeness
training/inference time
```

核心问题是：

> register embedding 越接近 full-set embedding，3DGS 质量是否越好？

如果答案是“基本相关”，这个研究方向就很有希望。

如果相关性弱，需要加入几何辅助 loss 或改变 readout 目标。

### 9.3 阶段 2：固定 K 训练 selector

先固定 K，例如：

```text
K = 10
K = 20
K = 10% * N
K = 20% * N
```

训练：

```text
L = L_nce + beta * L_pos
```

VGGT-OMEGA 冻结，readout head 和 selector 可训练。

### 9.4 阶段 3：可变 K 与最小集合

加入稀疏项：

```text
L = L_nce + beta * L_pos + lambda * |S|
```

调节 `lambda`，得到不同子集大小。

画 Pareto 曲线：

```text
x-axis: K/N
y-axis: PSNR, SSIM, LPIPS, register similarity
```

定义可接受阈值，例如：

```text
PSNR drop <= 1 dB
SSIM drop <= 0.02
LPIPS increase <= 0.02
```

在满足阈值的情况下选择最小 K。

### 9.5 阶段 4：端到端微调

只有在前面阶段有效后，才考虑端到端。

推荐顺序：

1. 冻结 VGGT-OMEGA，只训练 selector 和 readout。
2. 解冻 readout head。
3. 加 LoRA 或 adapter 到 VGGT-OMEGA 的后几层。
4. 最后才考虑全量 fine-tune。

全量端到端成本很高，也可能破坏 VGGT-OMEGA 原有几何能力，因此不应作为起点。

## 10. 数据组织建议

每个 training sample 应该是一个完整场景或一段视频：

```text
scene_id/
    images/
    optional_poses/
    optional_depths/
    optional_masks/
    metadata.json
```

建议按场景划分 train/val/test，不能把同一场景的不同帧拆到不同 split，否则会高估泛化能力。

需要覆盖：

- indoor / outdoor；
- object-centric / scene-centric；
- small baseline / wide baseline；
- static / dynamic；
- texture-rich / texture-poor；
- clean / motion blur；
- dense view / sparse view。

## 11. 可能的失败模式

### 11.1 选择语义多样，但几何不足

selector 可能偏向语义变化大的图像，但忽略视角基线、overlap、纹理和相机覆盖。解决方法是加入 pose diversity、matching coverage 或 3DGS proxy。

### 11.2 选择视角分散，但 overlap 不足

重建需要既有基线又有重叠。视角太分散会导致 SfM 或 3DGS 初始化困难。需要在 diversity 和 overlap 之间平衡。

### 11.3 register embedding 接近，但局部细节差

register 是全局表征，可能忽略局部细节。可以加入 patch-level coverage、depth confidence 或局部 feature matching loss。

### 11.4 selector 依赖 full VGGT tokens，节省不了 VGGT 推理成本

如果 selector 输入是完整集合的 VGGT register tokens，那么选择前已经运行过完整 VGGT-OMEGA。这个设置适合“离线压缩数据集，减少后续 3DGS/VLA 训练成本”，但不适合“降低 VGGT 推理成本”。如果想降低 VGGT 推理成本，需要使用更便宜的 image encoder 作为 selector 输入。

### 11.5 false negatives

InfoNCE 中不同场景可能很相似，非对角线不一定是真负样本。可以：

- 增大 batch 多样性；
- 避免同一地点或近似场景出现在同一 batch；
- 使用 soft labels；
- 加入正样本 cosine loss 稳定训练。

## 12. MVP 实现建议

第一版建议尽量简单。

### 12.1 模块

1. `FeatureCache`

   离线缓存：

   ```text
   z_full
   per-image features
   optional full-run depth/pose
   ```

2. `RegisterReadoutHead`

   仿照 `TextAlignmentHead`，输入 camera/register tokens，输出 normalized scene embedding。

3. `SetSelector`

   输入 per-image features，输出 per-image scores。

4. `SubsetSampler`

   训练时使用 Gumbel-TopK 或 straight-through top-K。

5. `TrainingLoop`

   固定 K，计算：

   ```text
   L_nce + beta * L_pos
   ```

### 12.2 第一版流程

```text
for each scene:
    run VGGT-OMEGA on dense/full image set
    cache z_full
    cache per-image feature x_i

for training:
    scores = selector({x_i})
    selected_indices = topK(scores, K)
    selected_images = images[selected_indices]
    z_sel = readout(VGGT-OMEGA(selected_images))
    loss = InfoNCE(z_sel, z_full) + beta * cosine_loss(z_sel, z_full)
```

### 12.3 第一版评估

对同一个 K，比较：

```text
random
uniform stride
DINO k-center
VGGT register k-center
learned selector
```

评价：

```text
register similarity
3DGS PSNR/SSIM/LPIPS
pose/depth metrics
time and memory
```

## 13. 参考文献与可借鉴方向

### 13.1 VGGT-OMEGA 与 registers

- VGGT-OMEGA paper: https://arxiv.org/abs/2605.15195
- VGGT-OMEGA project: https://vggt-omega.github.io/
- VGGT-OMEGA code: https://github.com/facebookresearch/vggt-omega
- Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588

启发：

- registers 可以作为全局信息载体；
- learnable token 可以读取 registers；
- VGGT-OMEGA 自身已经展示了 registers 对 VLA 和语言对齐有用。

### 13.2 Set 输入与 readout 网络

- Deep Sets: https://arxiv.org/abs/1703.06114
- Set Transformer: https://arxiv.org/abs/1810.00825
- Perceiver: https://arxiv.org/abs/2103.03206
- TokenLearner: https://arxiv.org/abs/2106.11297

启发：

- 输入图像集合是无序或弱有序 set；
- Set Transformer 和 Perceiver 适合处理可变数量输入；
- learnable latent/readout token 是合理的聚合方式。

### 13.3 对比学习与 InfoNCE

- Contrastive Predictive Coding: https://arxiv.org/abs/1807.03748
- CLIP: https://arxiv.org/abs/2103.00020
- SimCLR: https://arxiv.org/abs/2002.05709

启发：

- full-set embedding 与 subset embedding 可以作为正样本对；
- batch 内其他 scene 可以作为负样本；
- symmetric InfoNCE 适合做双向检索式对齐。

### 13.4 离散选择与可微采样

- Gumbel-Softmax: https://arxiv.org/abs/1611.01144
- Gumbel-Top-k: https://arxiv.org/abs/1903.06059

启发：

- top-K 选择可以用 Gumbel 近似或 straight-through estimator 训练；
- 比纯 RL 更适合作为第一版。

### 13.5 数据子集、视角选择与稀疏重建

- Skeletal Graphs/Sets for Efficient Structure from Motion: https://www.cs.cornell.edu/~snavely/projects/skeletalset/
- RegNeRF: https://arxiv.org/abs/2112.00724
- FSGS: https://arxiv.org/abs/2312.00451
- SparseGS: https://arxiv.org/abs/2312.00206
- GradMatch: https://arxiv.org/abs/2103.00123

启发：

- 传统 SfM 已经研究过如何从冗余图像集中选 skeletal subset；
- sparse-view NeRF/3DGS 说明少量视图重建是可行但困难的；
- 数据子集选择领域提供了 coverage、gradient matching、facility location 等思路。

## 14. 推荐结论

最推荐的研究路线是：

1. 不从 RL 开始。
2. 冻结 VGGT-OMEGA。
3. 先训练一个 fixed-K selector。
4. 使用 register readout embedding 做 full-set/subset distillation。
5. loss 从 `symmetric InfoNCE + cosine positive loss` 开始。
6. 用 3DGS 的 PSNR、SSIM、LPIPS 验证真实重建质量。
7. 建立 random、uniform、k-center、facility-location、pose coverage 等强 baseline。
8. 证明 register distance 与重建质量相关后，再加入 depth/pose/point 辅助 loss。
9. 最后再尝试可变 K、稀疏正则和端到端微调。

如果第一阶段能证明 learned selector 在相同 K 下稳定优于 uniform/random/k-center，并且在更小 K 下保持接近完整集合的 3DGS 质量，那么这个方向就有比较扎实的研究价值。
