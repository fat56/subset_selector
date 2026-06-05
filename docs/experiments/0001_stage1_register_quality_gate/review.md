# 复盘

## 决策

待定。准备流程、smoke 训练、bonsai 上使用 `stage1_split.json` 的 20% 30k FastGS sanity runs，以及 13 个 scene 的 random/uniform `images_4` 30k FastGS 矩阵都已经通过；但 register similarity 和相关性分析尚未补齐，当前不能据此通过或否决质量门。

## 证据

- 运行台账：[docs/registry/run_ledger.csv](../../registry/run_ledger.csv)
- 指标 schema：[docs/registry/metrics_schema.md](../../registry/metrics_schema.md)
- 当前结果记录：[results.md](results.md)
- FastGS rasterizer fix1 重编译日志：`runs/0001_stage1_register_quality_gate/fastgs_full_train/3dgsdata/mipnerf360_bonsai/rebuild_diff_gaussian_fix1.log`
- 已完成 30k sanity runs：bonsai full source、bonsai `random_ratio_seed000` 20%、bonsai `uniform_stride_ratio` 20%。
- 已完成 random/uniform `images_4` 30k 矩阵：78/78 done，0 failed，覆盖 13 个 scene x 5 random seed 以及 13 个 uniform stride run。
- Prepared source 评估语义修正：FastGS 已优先读取 `stage1_split.json`，避免原生 llffhold 对物化后的 source 重新切分。
- 正式矩阵的 split 校验：test split 先从 full image set 按 llffhold 计算，train split 再从非 test pool 中选择；队列脚本会拒绝 test set 不等于 full-scene llffhold 的 run。

## 下一步

- 保持 FastGS `diff-gaussian-rasterization_fastgs` 的本地 fix1，并在新环境中先跑一条 bonsai 30k sanity check。
- 保持 FastGS COLMAP reader 的 `stage1_split.json` 支持；prepared run 的正式指标必须使用该 split 口径。
- 缓存 full-set 与 subset 的 VGGT-OMEGA register/readout embedding。
- 汇总 register similarity 与 FastGS PSNR/SSIM/LPIPS 的 Spearman/Pearson，并检查 scene/seed 级失败样本。
- 生成或登记 feature/register per-image feature JSON 后，再启用 `feature_k_center` 和 `register_k_center`。
- 基于完整相关性、散点图和失败样本，再更新本复盘结论。
