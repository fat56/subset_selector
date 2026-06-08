# Review 记录

## Review 状态

已做阶段性 review。`hardlabel300` 2-layer 与 4-layer scale-up 已跑完，但 single unified learned readout 仍未通过 strict promotion gate。

## Review 问题

- readout target 是否使用 VGGT-native geometry consistency，而不是只用 appearance metrics？
- LTM30 是否保持为 held-out training 外数据？如果 split 有变化，是否明确记录？
- 训练后的 readout 在 scene-held-out validation 上是否优于 mean-pooled register cosine？
- direct sensor pose metrics 是否被排除在 pass/fail gate 之外？
- 选定的 readout checkpoint 是否在 `0004_stage2_fixed_k_selector_training` 开始前冻结？

## 风险

- 如果训练 scene 数太少，readout 可能过拟合。
- 基于 VGGT-native metrics 的 ranking labels 可能复现 VGGT 的内部偏差，而不是外部 metric geometry。
- soft embedding distance 仍可能和 hard subset VGGT rerun 不一致。
- 如果过早 joint training，高容量 readout 可能掩盖 selector 本身的失败。

## 当前建议

可以进入 `0004` 的 selector 设计，但默认使用 mean-pooled register cosine 作为保守 single-objective fallback。

当前最强 learned-readout 证据有两类：

- Single unified checkpoint: `hardlabel300_attention_multimetric_2layer/best_embedding.pt`，mean expected alignment `0.6063`，低于约 `0.6200` strict target。
- Metric-specific checkpoint set: retained per-target embedding mean `0.6210`，刚好过 gate，但不是单个 unified checkpoint。

除非 `0004` 明确采用 metric-specific readout losses，或先完成 embedding-combination evaluation，否则不 promotion learned readout。
