# 复盘

## 决策

待定。准备流程、smoke 训练和 bonsai 上使用 `stage1_split.json` 的 20% 30k FastGS sanity runs 已经通过，但完整跨 scene / 多 seed 质量矩阵和尚未补齐 register similarity，当前不能据此通过或否决质量门。

## 证据

- 运行台账：[docs/registry/run_ledger.csv](../../registry/run_ledger.csv)
- 指标 schema：[docs/registry/metrics_schema.md](../../registry/metrics_schema.md)
- 当前结果记录：[results.md](results.md)
- FastGS rasterizer fix1 重编译日志：`runs/0001_stage1_register_quality_gate/fastgs_full_train/3dgsdata/mipnerf360_bonsai/rebuild_diff_gaussian_fix1.log`
- 已完成 30k sanity runs：bonsai full source、bonsai `random_ratio_seed000` 20%、bonsai `uniform_stride_ratio` 20%。
- Prepared source 评估语义修正：FastGS 已优先读取 `stage1_split.json`，避免原生 llffhold 对 88 张物化图片重新切分。

## 下一步

- 保持 FastGS `diff-gaussian-rasterization_fastgs` 的本地 fix1，并在新环境中先跑一条 bonsai 30k sanity check。
- 保持 FastGS COLMAP reader 的 `stage1_split.json` 支持；prepared run 的正式指标必须使用该 split 口径。
- 重新跑完整 30k FastGS baseline 矩阵，并写入每个 run 的 `metrics.json`。
- 缓存 full-set 与 subset 的 VGGT-OMEGA register/readout embedding。
- 生成或登记 feature/register per-image feature JSON 后，再启用 `feature_k_center` 和 `register_k_center`。
- 汇总 Spearman/Pearson、散点图和失败样本，再更新本复盘结论。
