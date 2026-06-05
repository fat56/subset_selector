# Stage 1 Register Distance Versus FastGS Quality Gate

## Metadata

- Experiment ID: `0001_stage1_register_quality_gate`
- Stage: `stage1`
- Status: planned
- Created: 2026-06-03
- Config: [configs/experiments/0001_stage1_register_quality_gate.yaml](../../../configs/experiments/0001_stage1_register_quality_gate.yaml)

## Question

当每个 scene 只选取 20% 图片时，VGGT-OMEGA register/readout embedding similarity 是否与 sparse-view FastGS 重建质量显著正相关？

## Hypothesis

如果 register/readout embedding 捕获了足够的场景级几何信息，那么不同 baseline 在 20% ratio 预算下产生的子集，其 `register_cosine_similarity` 应该与 PSNR/SSIM 正相关、与 LPIPS 负相关。

## Method

先不训练 selector。对多个 scene、20% ratio 预算、多个 baseline 生成子集，计算：

- full set 与 subset 的 register/readout embedding similarity。
- 子集 FastGS 重建质量。
- 运行时间与子集比例。

首批 baseline：

- random ratio 20%
- uniform stride ratio 20%
- feature k-center
- register k-center

其中 feature/register k-center 需要预先提供对应 per-image feature JSON；没有 feature 缓存时先跑 random/uniform 作为 smoke pass。后续补入 FisherRF / InstantSplat co-visibility 等强 baseline。

FastGS source 目录由 `stage1-prepare` 生成：symlink 被选中的图片，并从 COLMAP text/binary model 过滤出对应的 `sparse/0` text model。

## Metrics

Primary:

- Spearman rho between `register_cosine_similarity` and `psnr`

Secondary:

- Pearson r
- SSIM / LPIPS correlation
- per-scene and per-method stability
- failure case count

## Decision Rule

通过建议：Spearman rho >= 0.5，且在 20% ratio 下多个 scene、多个 baseline 上方向稳定。

不通过：暂停 selector 训练，优先调整 readout 目标或加入几何辅助 proxy。
