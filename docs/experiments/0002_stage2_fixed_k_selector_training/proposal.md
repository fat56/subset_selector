# Stage 2 Fixed-K Selector Training

## Metadata

- Experiment ID: `0002_stage2_fixed_k_selector_training`
- Stage: `stage2`
- Status: design-draft
- Created: 2026-06-05
- Config: not created yet. Create only after this design is accepted.
- Depends on: `0001_stage1_register_quality_gate`

## Question

在 Stage 1 证明 register/readout embedding similarity 与 FastGS/3DGS 质量显著相关之后，能否训练一个固定预算 selector，在相同 `K` 或相同 ratio 下稳定选出比 random、uniform、feature k-center、register k-center 更好的图像子集？

## Hypothesis

如果 VGGT-OMEGA register/readout embedding 是有效的场景级几何 proxy，那么一个带上下文建模的 set selector 可以从每帧特征中学习到 coverage、overlap、清晰度和视角互补性，并在固定 `K` 下得到比手工规则更高的 hard-subset register similarity 与 FastGS 重建质量。

## Scope

本实验只设计和验证 Stage 2 的训练方案，不在当前步骤创建 config 或 `src` 实现。第一版固定预算，推荐从 Stage 1 已验证的 `20%` ratio 开始，再做 `10%`、`30%` 或固定 `K=10/20` 消融。

本实验不以节省 VGGT-OMEGA 推理为主要卖点。推荐 MVP 使用 Stage 1/Stage 2 cache 中的 VGGT per-image tokens 或 summaries 作为 selector 输入，适合离线数据集压缩。如果后续目标变成减少 VGGT 推理成本，应另开 cheap-image-feature selector，输入 DINO/CLIP/image-quality 特征。

## Recommended Architecture

### High-Level Flow

```text
full scene images
    -> frozen VGGT-OMEGA
    -> cached camera/register tokens
    -> locked RegisterReadoutHead
    -> z_full

per-image cached features
    -> SetSelector
    -> per-image score s_i
    -> relaxed top-K mask m_i for training
    -> soft-token proxy readout
    -> z_soft

topK(s_i)
    -> selected images sorted by original order
    -> frozen VGGT-OMEGA
    -> locked RegisterReadoutHead
    -> z_hard for validation
```

关键建议：第一版把 VGGT-OMEGA 冻结，把 Stage 1 通过 gate 的 readout head 也冻结。如果 Stage 1 还没有一个已锁定的 readout head，先补一个 `Stage 2.0 readout calibration`，用 full scene 和 dense subset augmentation 训练 readout，再冻结它训练 selector。不要一开始让 readout 和 selector 同时自由漂移，否则 `z_full` 目标会变动，正样本 cosine loss 容易失去约束力。

### Per-Image Feature

MVP 输入每张图一个 `x_i`：

```text
x_i = concat(
    camera_token_i,
    mean_pool(register_tokens_i),
    max_pool(register_tokens_i),
    optional_image_feature_i,
    scalar_features_i
)
```

推荐最小字段：

- `camera_token_i`: VGGT-OMEGA camera token。
- `register_mean_i`: 对该帧所有 register tokens 做 mean pooling。
- `register_max_i`: 对 register tokens 做 max pooling，补充显著局部响应。
- `frame_pos_i`: 原始帧序号归一化到 `[0, 1]`，只作为弱时间提示。
- `quality_i`: 可选，blur/texture/曝光等标量，缺失时置空并 mask。

如果有可靠 pose 或 matching/overlap cache，可以追加：

- camera center、view direction、relative baseline statistics。
- image matching degree、co-visibility score、depth confidence statistics。

### Feature Projector

```text
x_i
    -> LayerNorm
    -> Linear(d_in, 512)
    -> GELU
    -> Dropout(0.1)
    -> Linear(512, 512)
    -> LayerNorm
```

`d_model=512` 是第一版建议。显存紧张可降到 `256`，但不建议第一版低于 `256`，否则 set context 和 scoring head 容量可能不足。

### SetSelector Context Encoder

推荐第一版使用 Transformer/Set Transformer 风格 encoder：

```text
projected tokens [B, N, 512]
    -> 4 x Pre-LN TransformerEncoderBlock
       - Multi-head self-attention, 8 heads
       - FFN hidden dim 2048
       - dropout 0.1
       - padding mask for variable N
    -> contextual token h_i
```

设计理由：

- 子集选择依赖“这一帧相对其他帧是否冗余”，不能只用逐帧 MLP。
- fixed-K 场景里，分数必须是 scene-relative score，而不是全局质量分。
- Transformer encoder 足够直接，便于后续替换成 Perceiver 或 graph attention。

如果 `N` 很大，训练时先将每个 scene 限制到 `N_train_max=100-300` 的 dense teacher set。超过上限时用 uniform pre-sampling 或 Stage 1 强 baseline 预筛，验证时再看 full scene 版本。

### Score Head

```text
h_i
    -> LayerNorm
    -> Linear(512, 256)
    -> GELU
    -> Dropout(0.1)
    -> Linear(256, 1)
    -> score s_i
```

`s_i` 用于 soft relaxed top-K 训练和 hard top-K 验证。hard top-K 后必须按原始帧序排序再送入 VGGT-OMEGA，避免第一帧 token 初始化和输入顺序造成额外噪声。

### RegisterReadoutHead

readout 输入为 VGGT-OMEGA 的 camera/register tokens：

```text
tokens: [B, N, R + 1, C]
```

`R` 必须从 checkpoint/config 实读，不要写死 16。推荐结构：

```text
tokens
    -> reshape to [B, N * (R + 1), C]
    -> Linear(C, 512)
    -> concat 1 learnable readout token
    -> 2-4 x self-attention blocks
    -> take readout token
    -> Linear(512, 256 or 512)
    -> L2 normalize
```

第一版推荐 `1` 个 readout token、`2` 层 attention、输出维度 `256` 或 `512`。如果 Stage 1 显示单 embedding 对局部细节不敏感，再尝试 `4` 个 readout slots 后 pooling。

## Differentiability Plan

hard top-K 后重新运行 VGGT-OMEGA 是离散前向，loss 不能真实反传到“被选图片索引”。因此 Stage 2 第一版不要假装是完整端到端离散训练。推荐使用双路径：

### Training Path: Soft-Token Proxy

selector 输出 scores 后，用 relaxed top-K 得到连续 mask：

```text
m_i = RelaxedTopK(scores=s_i, k=K, temperature=tau)
sum_i m_i ~= K
0 <= m_i <= 1
```

然后在 cached full-run per-image tokens 上做 soft weighting：

```text
weighted_tokens_i = m_i * cached_tokens_i
z_soft = RegisterReadoutHead(weighted_tokens)
```

这个路径让 loss 可以反传到 selector，适合 warmup 和主训练。

### Validation Path: Hard Subset

定期执行：

```text
S = topK(s_i, K)
selected images = sort_by_original_order(S)
z_hard = RegisterReadoutHead(VGGT-OMEGA(selected images))
```

`z_hard` 是真正要看的指标来源。Stage 2 的 gate 以 hard subset 的 FastGS/3DGS 结果为准，而不是只看 `z_soft`。

### Optional Hard-Aware Refinement

如果 `z_soft` 很好但 `z_hard` 明显差，优先加候选子集 ranking/imitation，而不是立刻上 RL：

1. 对每个 scene 生成候选子集：random、uniform、k-center、register-k-center、Gumbel samples。
2. 用 hard VGGT-OMEGA similarity 或小规模 FastGS 指标给候选子集打分。
3. 训练 selector 让高分候选的 set score 高于低分候选：

```text
score(S) = mean_{i in S} s_i
L_rank = max(0, margin - score(S_good) + score(S_bad))
```

这个 refinement 不需要穿过离散 VGGT 前向，工程上比 policy gradient 稳定。

## Loss Design

本节是 loss menu，不是第一版全部启用的 recipe。Stage 2 的第一版目标应该尽量小，先验证 selector 能不能用最直接的 embedding distillation 学到有效选择策略。

推荐 MVP 从下面开始：

```text
L = L_pos
```

如果 batch size 足够且 scene 多样性较好，再打开 symmetric InfoNCE：

```text
L = w_pos * L_pos + w_nce * L_nce
```

只有当观察到具体失败模式时，再逐项加入辅助 loss：

```text
L = w_nce * L_nce
  + w_pos * L_pos
  + w_card * L_card
  + w_cov * L_cov
  + w_red * L_red
  + w_quality * L_quality
```

MVP 默认建议：

```text
w_pos = 1.0
w_nce = 0 or 1.0
tau_nce = 0.07
```

辅助项默认关闭：

```text
w_card = 1.0 if relaxed mask does not exactly sum to K, else 0
w_cov = 0 unless selected frames collapse to near-duplicates
w_red = 0 unless selected frames cluster too much
w_quality = 0 unless blur/quality failures dominate
```

设计原则：VGGT/VGGT-OMEGA 的训练 loss 用来学习几何 foundation model；这里的 selector loss 用来学习“选哪些图”。因此 coverage、redundancy、quality 这些项不是为了复刻 VGGT 训练，而是给 subset selection 的常见失败模式准备的补救约束。

### Symmetric InfoNCE

batch 中每个样本是一整个 scene：

```text
sim_ij = cosine(z_soft_i, stopgrad(z_full_j)) / tau_nce
L_nce = 0.5 * CE(sim, diag_labels) + 0.5 * CE(sim.T, diag_labels)
```

作用：让 subset embedding 匹配同 scene 的 full embedding，并避开 batch 内其他 scene。

### Positive Cosine Distillation

```text
L_pos = mean_i 1 - cosine(z_soft_i, stopgrad(z_full_i))
```

作用：稳定正样本对齐，避免 InfoNCE 被 false negative 或小 batch 噪声主导。`z_full` 建议 stop-gradient，尤其在 readout 未完全冻结时。

### Cardinality Loss

固定 `K` 时不需要稀疏 loss，但 relaxed mask 可能不严格等于 `K`：

```text
L_card = ((sum_i m_i - K) / K)^2
```

如果使用严格 relaxed top-K 且 `sum_i m_i = K`，这一项可以关掉。

### Coverage Loss

用 selector 输入特征或单独 coverage projection `c_i` 计算相似度，鼓励被选集合覆盖全 scene：

```text
Coverage = mean_j max_soft_i (log(m_i + eps) + cosine(c_i, c_j) / tau_cov)
L_cov = -Coverage
```

`max_soft` 可用 logsumexp 近似。它比单纯 diversity 更贴近“每张未选图都能被某张已选图代表”的目标。

### Redundancy Penalty

防止 top-K 全落在一段相似帧：

```text
L_red = sum_{i != j} m_i * m_j * relu(cosine(c_i, c_j) - delta) / K^2
```

第一版可先关掉，若观察到连续帧扎堆再启用。

### Quality Loss

如果有 blur/texture/depth-confidence 等质量分 `q_i`：

```text
L_quality = - sum_i m_i * q_i / K
```

这只是辅助项，权重必须小。否则 selector 可能只选清晰但几何覆盖不足的图。

### Geometry Auxiliary Loss

第一版不建议启用 depth/pose/point-map loss。原因是本阶段优先验证最核心假设：`z_subset` 接近 `z_full` 是否足以带来更好的重建子集。过早加入 depth loss 会把问题变成“用 VGGT depth pseudo-label 训练 selector”，同时引入 scale alignment、valid mask、动态区域、full/subset 上下文差异等额外复杂性。

只有当 Stage 2 的 embedding 指标改善但 FastGS 指标不跟随改善时，再加入几何辅助：

- depth log-scale distillation on selected frames。
- relative pose or pairwise camera distance consistency。
- point-map scale-aligned consistency。
- matching/overlap proxy loss。

## Training Schedule

1. `Stage 2.0 Readout Lock`，仅当 Stage 1 没有锁定 readout 时执行。训练 readout 区分不同 scene、对齐同 scene 的 full/dense subset，然后冻结。
2. `Stage 2.1 Soft Selector Warmup`，用 soft-token proxy 训练 selector。temperature 从 `1.0` 逐步降到 `0.2`。
3. `Stage 2.2 Hard Validation`，每隔固定 step 对 val scenes 做 hard top-K，重新跑冻结 VGGT-OMEGA，记录 `z_hard` 与 `z_full` 的 cosine。
4. `Stage 2.3 FastGS Validation`，对少量 val scenes 跑 FastGS/3DGS，比较 learned selector 和 Stage 1 baselines。
5. `Stage 2.4 Hard-Aware Refinement`，仅当 soft/hard gap 明显时启用候选子集 ranking。

## Metrics

Primary hard metrics:

- FastGS PSNR/SSIM/LPIPS at fixed `K` or fixed ratio。
- hard-subset `register_cosine_similarity = cosine(z_hard, z_full)`。
- win rate versus random/uniform/feature-k-center/register-k-center。

Training diagnostics:

- `z_soft` cosine vs `z_full`。
- soft-to-hard gap: `cos(z_soft, z_full) - cos(z_hard, z_full)`。
- retrieval top-1 accuracy under symmetric InfoNCE。
- selected-frame coverage score。
- selected indices distribution over time。
- duplicate/near-duplicate selection rate。

Efficiency:

- selector inference time。
- VGGT-OMEGA hard subset inference time。
- FastGS train time。
- subset size and ratio。

## Decision Rule

Stage 2 通过建议：

- 在 Stage 1 采用的主预算，例如 `20%` ratio，下 hard learned selector 的 median PSNR 至少优于 random 和 uniform。
- 强通过：median PSNR 优于 best non-learned baseline `>= 0.3 dB`，或 win rate `>= 60%`，同时 SSIM 不下降、LPIPS 不变差。
- hard-subset register cosine 至少不低于 best non-learned baseline，且 soft-to-hard gap 不持续扩大。
- 失败场景可解释，且不是由单一 scene 或单一 baseline 偶然主导。

不通过时：

- 如果 `z_soft` 好但 `z_hard` 差，优先加入 hard-aware ranking/imitation。
- 如果 hard register cosine 好但 FastGS 差，优先加入 coverage/geometry auxiliary，而不是继续加 selector 容量。
- 如果 learned selector 打不过 k-center/register-k-center，先保留 learned selector 作为 negative result，不进入 Stage 3。

## Risks

- `readout` 与 selector 同训导致目标漂移。缓解：先锁定 readout，`z_full` stop-gradient。
- soft-token proxy 与 hard subset 存在 gap。缓解：定期 hard validation，必要时加候选子集 ranking。
- selector 选连续高质量近邻帧。缓解：coverage loss、redundancy penalty、temporal distribution diagnostics。
- batch 内 false negatives。缓解：按场景多样性组 batch，使用 positive cosine loss 稳定训练。
- 依赖 full VGGT tokens 不能节省 VGGT 推理。缓解：在文档和实验命名中明确这是离线压缩路径。
