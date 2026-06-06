# 复盘

## 决策

当前不能通过 Stage 1 质量门。准备流程、smoke 训练、bonsai 上使用 `stage1_split.json` 的 20% 30k FastGS sanity runs，以及 13 个 scene 的 random/uniform `images_4` 30k FastGS 矩阵都已经通过；但 mean-pooled register-token proxy 与质量指标的 scene 内 Spearman 相关性偏弱，不能支持进入 selector 训练。

## 证据

- 运行台账：[docs/registry/run_ledger.csv](../../registry/run_ledger.csv)
- 指标 schema：[docs/registry/metrics_schema.md](../../registry/metrics_schema.md)
- 当前结果记录：[results.md](results.md)
- 指标复盘：[metric_review.md](metric_review.md)
- 几何指标结果：[geometry_metrics/geometry_correlation_summary.csv](geometry_metrics/geometry_correlation_summary.csv)
- FastGS rasterizer fix1 重编译日志：`runs/0001_stage1_register_quality_gate/fastgs_full_train/3dgsdata/mipnerf360_bonsai/rebuild_diff_gaussian_fix1.log`
- 已完成 30k sanity runs：bonsai full source、bonsai `random_ratio_seed000` 20%、bonsai `uniform_stride_ratio` 20%。
- 已完成 random/uniform `images_4` 30k 矩阵：78/78 done，0 failed，覆盖 13 个 scene x 5 random seed 以及 13 个 uniform stride run。
- Prepared source 评估语义修正：FastGS 已优先读取 `stage1_split.json`，避免原生 llffhold 对物化后的 source 重新切分。
- 正式矩阵的 split 校验：test split 先从 full image set 按 llffhold 计算，train split 再从非 test pool 中选择；队列脚本会拒绝 test set 不等于 full-scene llffhold 的 run。
- 已完成 VGGT-OMEGA register-token cache：13 个 full-train(non-test) reference + 78 个 random/uniform subset，共 91/91 成功。
- `register_mean_cosine` scene 内相关性：PSNR mean Spearman 0.2088，SSIM mean Spearman 0.2352，LPIPS mean Spearman -0.2879；best-cosine 与 best-quality 只在 4/13 个 scene 上一致。
- 已补 point-cloud 几何 proxy：`colmap_sparse_full_scene` 覆盖 13 个 scene；`fastgs_full_train_images4` 目前只覆盖 bonsai。COLMAP sparse proxy 上，F-score@1% mean Spearman 0.0462，Chamfer-L1 mean Spearman -0.0418，accuracy mean Spearman -0.0198，completeness mean Spearman -0.1209，仍不足以通过质量门。
- 详细表：[register_similarity/subset_register_similarity.csv](register_similarity/subset_register_similarity.csv) 和 [register_similarity/scene_register_correlations.csv](register_similarity/scene_register_correlations.csv)。

## 下一步

- 保持 FastGS `diff-gaussian-rasterization_fastgs` 的本地 fix1，并在新环境中先跑一条 bonsai 30k sanity check。
- 保持 FastGS COLMAP reader 的 `stage1_split.json` 支持；prepared run 的正式指标必须使用该 split 口径。
- 不基于 mean-pooled register-token proxy 进入 Stage 2 selector 训练。
- 训练或校准一个 readout head，或改用更强 VGGT-native geometry proxy 后，重新计算 scene 内相似度与 FastGS PSNR/SSIM/LPIPS 及 point-cloud geometry metrics 的关系。
- 补齐全 13 个 scene 的 full-train FastGS pseudo-GT，或从 rendered depth fusion / surface samples 计算几何指标，减少 raw Gaussian centers 与 sparse COLMAP reference 的偏差。
- 重新缓存 VGGT-Omega depth/pose/point-map 后，加入 depth/normal/point-map consistency。
- 生成或登记 feature/register per-image feature JSON 后，再启用 `feature_k_center` 和 `register_k_center`。
- 基于完整相关性、散点图和失败样本，再更新本复盘结论。
