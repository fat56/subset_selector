# Roadmap

## Stage 0: Baseline And Evaluation Pipeline

目标：先建强 baseline 和 sparse-view 3DGS 评测流程，不训练 selector。

必须产出：

- random/uniform/k-center/register-k-center 等基础方法的 selected indices。
- 每个子集的 register/readout similarity。
- 3DGS 或 InstantSplat 评测指标：PSNR、SSIM、LPIPS、pose/depth 可选指标。
- 每次 run 的 manifest、metrics 和 ledger 记录。

## Stage 1: Register Quality Gate

核心问题：

```text
register cosine similarity 与 3DGS PSNR/SSIM/LPIPS 是否显著相关？
```

建议通过标准：

- Spearman rho >= 0.5。
- 多个 K、多种 baseline、多个 scene 上方向稳定。
- 不只看均值，也检查失败场景。

不通过时：先加几何辅助目标、换 readout 目标或重定 proxy，不进入 Stage 2。

## Stage 2: Fixed-K Selector

前置条件：Stage 1 gate 通过。

目标：固定 K，训练 selector + readout，VGGT-OMEGA 冻结。GATE 是同 K 下稳定优于 random/uniform/k-center/FisherRF。

## Stage 3: Variable-K And Pareto Curve

加入稀疏项或 stopping policy，画 K/N 与质量指标的 Pareto 曲线，找最小可接受子集。

## Stage 4: Careful End-To-End Adaptation

仅在前面阶段有效时推进。顺序建议：readout 解冻，VGGT-OMEGA 后层 LoRA/adapter，最后才考虑全量微调。

