# Stage 1 Register Distance Versus 3DGS Quality Gate

## Metadata

- Experiment ID: `0001_stage1_register_quality_gate`
- Stage: `stage1`
- Status: planned
- Created: 2026-06-03
- Config: [configs/experiments/0001_stage1_register_quality_gate.yaml](../../../configs/experiments/0001_stage1_register_quality_gate.yaml)

## Question

VGGT-OMEGA register/readout embedding similarity 是否与 sparse-view 3DGS 重建质量显著正相关？

## Hypothesis

如果 register/readout embedding 捕获了足够的场景级几何信息，那么不同 baseline 和不同 K 产生的子集，其 `register_cosine_similarity` 应该与 PSNR/SSIM 正相关、与 LPIPS 负相关。

## Method

先不训练 selector。对多个 scene、多个预算 K、多个 baseline 生成子集，计算：

- full set 与 subset 的 register/readout embedding similarity。
- 子集 3DGS 或 InstantSplat 重建质量。
- 运行时间与子集比例。

首批 baseline：

- random K
- uniform stride K
- feature k-center
- register k-center

后续补入 FisherRF / InstantSplat co-visibility 等强 baseline。

## Metrics

Primary:

- Spearman rho between `register_cosine_similarity` and `psnr`

Secondary:

- Pearson r
- SSIM / LPIPS correlation
- per-scene and per-K stability
- failure case count

## Decision Rule

通过建议：Spearman rho >= 0.5，且在多个 K、多个 scene、多个 baseline 上方向稳定。

不通过：暂停 selector 训练，优先调整 readout 目标或加入几何辅助 proxy。

