# Review 记录

## Review 状态

尚未 review。

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

继续推进设计和 dataset building。在 readout 通过 gate，或 selector 实验明确选择 mean pooling 作为 baseline objective 之前，不启动 fixed-K selector training。
