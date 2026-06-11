# 0006 Stage2 Step-Gain Teacher

## 目标

0005 已证明 true swap-gain candidate 本身有 VGGT-native 正信号，但 image-only student 在 candidate top1 / gate 设定下泛化不稳。本实验把监督改成更接近 step-level marginal gain：给定当前 fixed-K base subset `S`，评估单步替换 `S - r + i` 是否提升 VGGT-native teacher score。

第一版不直接做完整 greedy tree，而是使用 `uniform20` 作为 base `S0`，生成更密的 single-swap candidates：

```text
S0 = uniform20(scene)
for candidate frame i not in S0:
  remove nearest selected frame r in DINO feature space
  evaluate S0 - r + i with VGGT-native metrics
  gain_i = target_error(S0) - target_error(S0 - r + i)
```

这样每条 label 都可以被解释为“把某张非 uniform 帧加入当前集合的边际收益”，比 0005 中每个 scene 只有 4 个 swap candidates 更适合训练 student 判断是否值得替换。

## 边界

Student 推理输入仍然只允许使用 image-only features：DINOv2-S/ViT-S global embedding、patch summary、image stats、temporal stats。VGGT-OMEGA 只用于离线 teacher label，不作为 student input。

## Main V1 计划

1. 复用 0005 richer labels 和 existing full/uniform cache。
2. 每个 scene 生成 `8` 个 single-swap candidates，命名为 `stepgain20_dino1_rank000-007` 或复用 `swapgain20_dino1_rank000-007`。
3. 对这些 candidates 跑 VGGT-native metrics，并合并到 0005 augmented labels。
4. 先训练现有 explicit gate head，判断 dense single-swap label 是否比 0005 的 sparse swap-gain 更稳定。
5. 若 gate 仍不稳，再训练真正的 frame-level gain student：输入每帧 image-only feature，回归 `gain_i`，用 top gain replacements 形成 fixed-K subset。

## 判定

通过条件：

- held-out test `uniform_minus_learned_error > 0`，并且不只出现在单个 seed。
- val-selected rule 在 test 上仍为正，不依赖 test-oracle threshold。
- student 不读取任何 VGGT token/output。

不通过条件：

- val 全正但 test 多数为负，视为 calibration failure。
- test 最佳规则仍是永远回到 `uniform20`。
- dense single-swap labels 只让模型更保守，test 接近 `0.0000`。

## 资源预估

以 300 scenes、每 scene 8 个 single swaps 估算：

- VGGT cache jobs: `2400`
- 磁盘: 约 `100G` 量级
- 当前剩余磁盘约 `441G`，足够第一版实验。

