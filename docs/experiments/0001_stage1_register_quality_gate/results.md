# 结果

第一阶段准备流程、FastGS random/uniform baseline 矩阵，以及 mean-pooled register-token similarity 分析已经完成；训练式 readout head 和更强 baseline 仍待补齐。

- VGGT-OMEGA 严格预检：通过。
- FastGS 严格预检：通过。
- `stage1-prepare --dataset 3dgsdata --overwrite`：已完成。
- 已生成 manifest：共 104 个。
- 可运行 FastGS random/uniform runs：78 个，覆盖 13 个 scene x 5 个 random seed，以及 uniform stride；`images_4` 30k 矩阵已完成 78/78，失败 0。
- 待补齐 runs：26 个，原因是 `feature_k_center` 和 `register_k_center` 需要先在 `data/datasets.yaml` 中登记 feature JSON。
- 默认分辨率长训触发 CUDA rasterizer 失败后，FastGS 命令脚本已用 `--resolution 800` 重新生成。
- FastGS `diff-gaussian-rasterization_fastgs` 已应用本地 `fix1` 并用 CUDA 12.8 / `TORCH_CUDA_ARCH_LIST=12.0` 重编译；bonsai full source 和两个 20% prepared source 已完成 30k 训练及评估。
- FastGS COLMAP reader 已补充 `stage1_split.json` 支持；prepared source 开启 `--eval` 时会使用 Stage 1 固定的 train/test 划分，而不是对物化后的 source 重新执行 llffhold。
- 正式 random/uniform 矩阵使用 cropped/downsampled `images_4`，`--iterations 30000`，`--densification_interval 100`，FastGS 默认 `resolution=-1`。
- VGGT-OMEGA register-token cache 已完成：13 个 full-train(non-test) reference + 78 个 random/uniform subset，共 91/91 成功。

## 运行记录

| 运行 ID | 方法 | 数据集 | Ratio | Register Sim | PSNR | SSIM | LPIPS | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| prepared/3dgsdata/mipnerf360_bonsai/random_ratio_seed000/ratio_020/fastgs_output_probe_res800_3k | random_ratio_seed000 | 3dgsdata/mipnerf360_bonsai | 0.20 | pending | 27.8000 | 0.9062 | 0.1643 | FastGS 3k-iteration smoke，`--resolution 800`，11 张 held-out test view。不是质量门正式 run。 |
| fastgs_full_train/3dgsdata/mipnerf360_bonsai/full_train_eval_res800_30k_retry | full_train | 3dgsdata/mipnerf360_bonsai | 1.00 | n/a | failed | failed | failed | 原始 source、不选图、`--eval --resolution 800 --iterations 30000`；292 张图按 llffhold=8 拆分为约 255 train / 37 test，约 1,050 iteration 触发 CUDA illegal memory access。日志见 `runs/0001_stage1_register_quality_gate/fastgs_full_train/3dgsdata/mipnerf360_bonsai/full_train_eval_res800_30k_retry/train.log`。 |
| fastgs_full_train/3dgsdata/mipnerf360_bonsai/full_train_eval_res800_30k_cuda_blocking | full_train_debug | 3dgsdata/mipnerf360_bonsai | 1.00 | n/a | failed | failed | failed | 同上，加 `CUDA_LAUNCH_BLOCKING=1`；约 1,450 iteration 在 `diff_gaussian_rasterization_fastgs._C.rasterize_gaussians` forward 内触发 CUDA illegal memory access。日志见 `runs/0001_stage1_register_quality_gate/fastgs_full_train/3dgsdata/mipnerf360_bonsai/full_train_eval_res800_30k_cuda_blocking/train.log`。 |
| fastgs_full_train/3dgsdata/mipnerf360_bonsai/full_train_images4_vfm5090_baseline_30k | full_train_debug | 3dgsdata/mipnerf360_bonsai | 1.00 | n/a | failed | failed | failed | 原始 source、`--images images_4 --eval --iterations 30000 --densification_interval 500 --highfeature_lr 0.02 --grad_abs_thresh 0.0006`；未应用 fix1 时约 3,450 iteration 在 rasterizer forward 内触发 CUDA illegal memory access。 |
| fastgs_full_train/3dgsdata/mipnerf360_bonsai/full_train_images4_vfm5090_fix1_30k | full_train_fix1 | 3dgsdata/mipnerf360_bonsai | 1.00 | n/a | 32.4994 | 0.9498 | 0.1110 | 应用并重编译 FastGS rasterizer fix1 后，full source `images_4` 30k 训练完成；37 张 held-out test view。`metrics.json` 已写入该 run。 |
| prepared/3dgsdata/mipnerf360_bonsai/random_ratio_seed000/ratio_020/fastgs_output_res800_30k_fix1 | random_ratio_seed000_diag | 3dgsdata/mipnerf360_bonsai | 0.20 | pending | 29.4773 | 0.9327 | 0.1155 | 诊断 run：应用 fix1 后可完成 30k，但 FastGS 原生 `--eval` 对物化后的 88 张图重新 llffhold，实际评估 11 张 test view；不作为质量门正式口径。 |
| prepared/3dgsdata/mipnerf360_bonsai/uniform_stride_ratio/ratio_020/fastgs_output_res800_30k_fix1 | uniform_stride_ratio_diag | 3dgsdata/mipnerf360_bonsai | 0.20 | pending | 29.4434 | 0.9306 | 0.1191 | 诊断 run：同上，实际评估 11 张 test view；不作为质量门正式口径。 |
| prepared/3dgsdata/mipnerf360_bonsai/random_ratio_seed000/ratio_020/fastgs_output_res800_30k_fix1_stage1split | random_ratio_seed000 | 3dgsdata/mipnerf360_bonsai | 0.20 | pending | 28.1985 | 0.9171 | 0.1313 | 正式 Stage 1 split run：FastGS 使用 `stage1_split.json`，51 train / 37 held-out test views，训练耗时 70.58s。`metrics.json` 已写入该 run。 |
| prepared/3dgsdata/mipnerf360_bonsai/uniform_stride_ratio/ratio_020/fastgs_output_res800_30k_fix1_stage1split | uniform_stride_ratio | 3dgsdata/mipnerf360_bonsai | 0.20 | pending | 28.6207 | 0.9186 | 0.1229 | 正式 Stage 1 split run：FastGS 使用 `stage1_split.json`，51 train / 37 held-out test views，训练耗时 71.95s。`metrics.json` 已写入该 run。 |
| prepared/3dgsdata/*/random_ratio_seed000/ratio_020/fastgs_output_images4_30k | random_ratio_seed000 | 3dgsdata | 0.20 | see CSV | 22.3184 | 0.6863 | 0.2673 | 13 条正式 Stage 1 split run 均完成；表中仅为台账均值，不作为相关性判断。register similarity 只按 scene 内比较，详见 CSV。 |
| prepared/3dgsdata/*/random_ratio_seed001/ratio_020/fastgs_output_images4_30k | random_ratio_seed001 | 3dgsdata | 0.20 | see CSV | 22.0161 | 0.6757 | 0.2732 | 13 条正式 Stage 1 split run 均完成；表中仅为台账均值，不作为相关性判断。register similarity 只按 scene 内比较，详见 CSV。 |
| prepared/3dgsdata/*/random_ratio_seed002/ratio_020/fastgs_output_images4_30k | random_ratio_seed002 | 3dgsdata | 0.20 | see CSV | 22.3021 | 0.6887 | 0.2661 | 13 条正式 Stage 1 split run 均完成；表中仅为台账均值，不作为相关性判断。register similarity 只按 scene 内比较，详见 CSV。 |
| prepared/3dgsdata/*/random_ratio_seed003/ratio_020/fastgs_output_images4_30k | random_ratio_seed003 | 3dgsdata | 0.20 | see CSV | 22.1264 | 0.6858 | 0.2688 | 13 条正式 Stage 1 split run 均完成；表中仅为台账均值，不作为相关性判断。register similarity 只按 scene 内比较，详见 CSV。 |
| prepared/3dgsdata/*/random_ratio_seed004/ratio_020/fastgs_output_images4_30k | random_ratio_seed004 | 3dgsdata | 0.20 | see CSV | 22.2143 | 0.6813 | 0.2699 | 13 条正式 Stage 1 split run 均完成；表中仅为台账均值，不作为相关性判断。register similarity 只按 scene 内比较，详见 CSV。 |
| prepared/3dgsdata/*/uniform_stride_ratio/ratio_020/fastgs_output_images4_30k | uniform_stride_ratio | 3dgsdata | 0.20 | see CSV | 22.7165 | 0.6978 | 0.2583 | 13 条正式 Stage 1 split run 均完成；表中仅为台账均值，不作为相关性判断。register similarity 只按 scene 内比较，详见 CSV。 |

## Random/Uniform `images_4` 30k 矩阵

2026-06-05 完成 `data/raw/3dgsdata` 上 random/uniform FastGS 矩阵。队列状态为 `jobs_total=78 done=78 failed=0`，输出路径为每个 prepared run 下的 `fastgs_output_images4_30k/results.json`。

运行口径：

- 训练图片使用 cropped/downsampled `images_4`；对没有原生 `images_4` 的 source，由队列脚本生成 factor-4 图片目录。
- FastGS 参数为 `--eval --images images_4 --iterations 30000 --densification_interval 100`，模型配置记录的 `resolution=-1`。
- 测试集先从 full image set 按 full-scene llffhold 切出；train set 再从非 test pool 中选取。`scripts/run_fastgs_random_uniform_queue.sh prepare` 会校验每个 run 有 `stage1_split.json`、`test_images` 等于 full-scene llffhold split、`train_images` 等于 selected subset，且 train/test 不相交。

下表只作为 FastGS 矩阵运行概览，不用于判断 register similarity 是否有效。质量门相关性必须在同一个 scene 内比较 5 个 random 和 1 个 uniform，不把不同 scene 的同名 random seed 直接混成一个判断统计。

| Dataset | Method | Runs | PSNR mean | SSIM mean | LPIPS mean |
|---|---|---:|---:|---:|---:|
| mipnerf360 | random_seed000 | 9 | 22.7658 | 0.6578 | 0.2872 |
| mipnerf360 | random_seed001 | 9 | 22.4022 | 0.6414 | 0.2963 |
| mipnerf360 | random_seed002 | 9 | 22.6333 | 0.6583 | 0.2869 |
| mipnerf360 | random_seed003 | 9 | 22.3079 | 0.6515 | 0.2934 |
| mipnerf360 | random_seed004 | 9 | 22.4582 | 0.6470 | 0.2932 |
| mipnerf360 | uniform | 9 | 23.0551 | 0.6639 | 0.2821 |
| tandt | random_seed000 | 2 | 20.4199 | 0.7757 | 0.1662 |
| tandt | random_seed001 | 2 | 19.9478 | 0.7589 | 0.1788 |
| tandt | random_seed002 | 2 | 20.4396 | 0.7712 | 0.1705 |
| tandt | random_seed003 | 2 | 20.3936 | 0.7723 | 0.1692 |
| tandt | random_seed004 | 2 | 20.7055 | 0.7775 | 0.1639 |
| tandt | uniform | 2 | 20.6647 | 0.7947 | 0.1511 |
| db | random_seed000 | 2 | 22.2037 | 0.7250 | 0.2787 |
| db | random_seed001 | 2 | 22.3469 | 0.7471 | 0.2636 |
| db | random_seed002 | 2 | 22.6742 | 0.7426 | 0.2681 |
| db | random_seed003 | 2 | 23.0429 | 0.7533 | 0.2577 |
| db | random_seed004 | 2 | 22.6253 | 0.7398 | 0.2711 |
| db | uniform | 2 | 23.2447 | 0.7538 | 0.2586 |

PSNR 上，`mipnerf360` 和 `db` 组最高的是 `uniform`；`tandt` 组最高的是 `random_seed004`。LPIPS 越低越好。

按 scene 的 PSNR 分列如下，便于后续排查 seed 差异：

| Scene | Dataset | Train/Test | random_seed000 | random_seed001 | random_seed002 | random_seed003 | random_seed004 | uniform |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| db_drjohnson | db | 46/33 | 22.2870 | 22.7602 | 21.9373 | 23.2168 | 21.9359 | 22.7489 |
| db_playroom | db | 40/29 | 22.1204 | 21.9337 | 23.4112 | 22.8690 | 23.3146 | 23.7406 |
| mipnerf360_bicycle | mipnerf360 | 34/25 | 20.5653 | 20.1063 | 20.2854 | 20.8410 | 18.9628 | 19.7752 |
| mipnerf360_bonsai | mipnerf360 | 51/37 | 27.9375 | 27.7863 | 27.6719 | 26.8998 | 27.4355 | 28.9883 |
| mipnerf360_counter | mipnerf360 | 42/30 | 24.6960 | 24.1527 | 24.5797 | 23.1054 | 24.5898 | 25.6509 |
| mipnerf360_flowers | mipnerf360 | 31/22 | 16.3682 | 15.9761 | 16.5984 | 16.1625 | 15.5813 | 16.1695 |
| mipnerf360_garden | mipnerf360 | 33/24 | 23.1380 | 22.8501 | 23.3772 | 23.0876 | 23.3056 | 23.8380 |
| mipnerf360_kitchen | mipnerf360 | 49/35 | 27.2814 | 26.7169 | 27.6026 | 26.6458 | 27.6068 | 28.1889 |
| mipnerf360_room | mipnerf360 | 55/39 | 28.1515 | 27.8349 | 26.9928 | 26.8234 | 27.2224 | 27.7214 |
| mipnerf360_stump | mipnerf360 | 22/16 | 19.5110 | 19.2803 | 19.2804 | 19.7165 | 19.6868 | 19.7489 |
| mipnerf360_treehill | mipnerf360 | 25/18 | 17.2431 | 16.9159 | 17.3109 | 17.4889 | 17.7328 | 17.4147 |
| tandt_train | tandt | 53/38 | 18.1991 | 17.5700 | 17.9895 | 18.0281 | 18.4011 | 18.2481 |
| tandt_truck | tandt | 44/32 | 22.6406 | 22.3255 | 22.8897 | 22.7590 | 23.0099 | 23.0813 |

## 相关性

本轮只评估一个透明 proxy：`register_mean_cosine`。具体做法是对 VGGT-OMEGA `register_tokens` 在 batch、frame、register 维度做 mean pooling，得到一个 scene/subset 向量；每个 subset 与同 scene 的 full-train(non-test) reference 向量计算 cosine。它不是训练后的 readout head。

指标复盘见 [metric_review.md](metric_review.md)。核心结论是：PSNR/SSIM/LPIPS 对 appearance fidelity 有价值，但不是 register token 的最佳目标；后续质量门应优先加入 point-cloud F-score、accuracy/completeness、Chamfer-L1/L2，以及 full-train pseudo-GT 或 VGGT-native depth/pose/point-map consistency。

输入口径：

- full reference：每个 scene 的 full image set 先按 full-scene llffhold 排除 test，再把剩余 train candidates 输入 VGGT-OMEGA。
- subset：同一 scene 内 5 个 `random_ratio_seed000-004` 和 1 个 `uniform_stride_ratio` 的 `stage1_split.json/train_images`。
- VGGT-OMEGA：checkpoint `vggt_omega_1b_512.pt`，`image_resolution=512`，`mode=balanced`。
- 详细 CSV：[subset_register_similarity.csv](register_similarity/subset_register_similarity.csv)，[scene_register_correlations.csv](register_similarity/scene_register_correlations.csv)。

方向统计如下。每个 scene 先在 6 个候选子集内计算一次相关性，再对 13 个 scene 汇总；不把不同 scene 的 subset 样本直接混在一起。

| Metric | Expected direction | Mean Spearman | Spearman sign | Mean Pearson | Pearson sign |
|---|---|---:|---:|---:|---:|
| PSNR | positive | 0.2088 | 9/13 | 0.2936 | 12/13 |
| SSIM | positive | 0.2352 | 8/13 | 0.3876 | 13/13 |
| LPIPS | negative | -0.2879 | 10/13 | -0.4423 | 13/13 |

按数据集组汇总 scene-level Spearman：

| Dataset | Scenes | PSNR mean rho | PSNR sign | SSIM mean rho | SSIM sign | LPIPS mean rho | LPIPS sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| mipnerf360 | 9 | 0.2317 | 6/9 | 0.2381 | 5/9 | -0.2825 | 6/9 |
| tandt | 2 | 0.2286 | 2/2 | 0.2286 | 2/2 | -0.2857 | 2/2 |
| db | 2 | 0.0857 | 1/2 | 0.2286 | 1/2 | -0.3143 | 2/2 |

按 scene 的相关性：

| Scene | PSNR rho/r | SSIM rho/r | LPIPS rho/r | Best cosine | Best PSNR |
|---|---:|---:|---:|---|---|
| db_drjohnson | 0.5429 / 0.4198 | 0.5429 / 0.4491 | -0.6000 / -0.3628 | uniform | random_seed003 |
| db_playroom | -0.3714 / 0.2927 | -0.0857 / 0.5878 | -0.0286 / -0.5461 | random_seed001 | uniform |
| mipnerf360_bicycle | -0.0286 / 0.2322 | -0.0286 / 0.1595 | 0.0286 / -0.2774 | random_seed001 | random_seed003 |
| mipnerf360_bonsai | 0.0857 / 0.3058 | -0.0857 / 0.1598 | 0.1429 / -0.1493 | random_seed001 | uniform |
| mipnerf360_counter | 0.0857 / 0.3660 | 0.2000 / 0.3643 | -0.3143 / -0.3717 | random_seed001 | uniform |
| mipnerf360_flowers | -0.0286 / 0.3758 | -0.0286 / 0.2779 | -0.0286 / -0.3681 | random_seed001 | random_seed002 |
| mipnerf360_garden | 0.9429 / 0.7955 | 0.7143 / 0.6437 | -0.8857 / -0.8083 | uniform | uniform |
| mipnerf360_kitchen | 0.4286 / 0.4684 | 0.6571 / 0.7107 | -0.8286 / -0.7739 | uniform | uniform |
| mipnerf360_room | 0.2571 / 0.5334 | -0.0857 / 0.3651 | -0.2000 / -0.7143 | uniform | random_seed000 |
| mipnerf360_stump | 0.3714 / 0.0413 | 0.4857 / 0.3495 | -0.4857 / -0.2966 | uniform | uniform |
| mipnerf360_treehill | -0.0286 / 0.0383 | 0.3143 / 0.3393 | 0.0286 / -0.3967 | uniform | random_seed004 |
| tandt_train | 0.1429 / -0.1064 | 0.0857 / 0.3701 | -0.2571 / -0.3562 | uniform | random_seed004 |
| tandt_truck | 0.3143 / 0.0545 | 0.3714 / 0.2621 | -0.3143 / -0.3286 | uniform | uniform |

质量门结论：mean-pooled register token proxy 有一定方向性，尤其 Pearson 的方向较稳定，但 Spearman 均值远低于通过建议阈值 0.5，且 best-cosine 与 best-quality 只在 4/13 个 scene 上一致。因此不能基于这个 proxy 通过 Stage 1；下一步应补训练式 readout 或更强几何 proxy，再和 feature/register k-center 等 baseline 一起复核。

## 观察

- `stage1-prepare` 现在会把选中的 train images 和 llffhold held-out test images 一起物化到每个 FastGS source，并写入 split metadata。
- `mipnerf360_bonsai/random_ratio_seed000` 的默认 FastGS 长训在约 2,800 iteration 处失败，错误为 `diff_gaussian_rasterization_fastgs` 中的 CUDA illegal memory access。
- 同一 run 使用 `--resolution 800` 后通过了 3k probe，并成功渲染、评估 held-out test views；但完整 30k run 仍在约 14,960 iteration 处于同一路 rasterizer 路径失败。
- 一个更保守的 30k probe 使用 `--resolution 800` 并把 densification 推迟到训练结束之后，但仍在约 4,360 iteration 处因 rasterizer CUDA address-space error 失败。
- 按原始 `data/raw/3dgsdata/mipnerf360/bonsai` source 直接 full-train、不选图的 30k 诊断也不能完成；普通 run 在约 1,050 iteration 失败，同步调试 run 在约 1,450 iteration 失败。同步栈显示错误发生在 `diff_gaussian_rasterization_fastgs._C.rasterize_gaussians` forward 内部。
- 进一步排查发现未应用 fix1 的 `images_4` full-train baseline 在约 3,450 iteration 仍会触发同一路 CUDA illegal memory access。
- 本地 FastGS rasterizer fix1 做了三类保护：过滤低 opacity 或非法 compact-box 的 splat，初始化未写满的 tile key/list，并在 tile range 写入前检查 tile id 边界。
- fix1 重编译后，bonsai full source、`random_ratio_seed000` 20% source、`uniform_stride_ratio` 20% source 均完成 30k 训练、test render 和 metrics 评估。
- FastGS 原生 COLMAP `--eval` 会对当前 source 内全部图片重新按 llffhold 切分；对 Stage 1 prepared source 这会把部分 held-out 图片混回训练集。已在 FastGS `scene/dataset_readers.py` 中加入 `stage1_split.json` 优先逻辑，并重跑 random/uniform 两条 20% sanity run。
- random/uniform `images_4` 30k 矩阵稳定完成；mean-pooled register-token similarity 已补齐，但不足以通过质量门，当前 blocker 转为训练/校准 readout 或更换几何 proxy，并补齐 feature/register k-center。
