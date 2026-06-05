# 结果

第一阶段准备流程和 FastGS random/uniform baseline 矩阵已经完成，register/readout 相关性仍待补齐。

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
| prepared/3dgsdata/*/random_ratio_seed000-004/ratio_020/fastgs_output_images4_30k | random_ratio_seed000-004 | 3dgsdata | 0.20 | pending | 22.1954 | 0.6836 | 0.2691 | 65 条正式 Stage 1 split run 均完成；表中为所有 random seed pooled mean。使用 `images_4`、30k iterations、densification interval 100。 |
| prepared/3dgsdata/*/uniform_stride_ratio/ratio_020/fastgs_output_images4_30k | uniform_stride_ratio | 3dgsdata | 0.20 | pending | 22.7165 | 0.6978 | 0.2583 | 13 条正式 Stage 1 split run 均完成；表中为 scene mean。使用 `images_4`、30k iterations、densification interval 100。 |

## Random/Uniform `images_4` 30k 矩阵

2026-06-05 完成 `data/raw/3dgsdata` 上 random/uniform FastGS 矩阵。队列状态为 `jobs_total=78 done=78 failed=0`，输出路径为每个 prepared run 下的 `fastgs_output_images4_30k/results.json`。

运行口径：

- 训练图片使用 cropped/downsampled `images_4`；对没有原生 `images_4` 的 source，由队列脚本生成 factor-4 图片目录。
- FastGS 参数为 `--eval --images images_4 --iterations 30000 --densification_interval 100`，模型配置记录的 `resolution=-1`。
- 测试集先从 full image set 按 full-scene llffhold 切出；train set 再从非 test pool 中选取。`scripts/run_fastgs_random_uniform_queue.sh prepare` 会校验每个 run 有 `stage1_split.json`、`test_images` 等于 full-scene llffhold split、`train_images` 等于 selected subset，且 train/test 不相交。

汇总指标：

| 方法 | Runs | PSNR mean | SSIM mean | LPIPS mean | 备注 |
|---|---:|---:|---:|---:|---|
| random_ratio_seed000-004 | 65 | 22.1954 | 0.6836 | 0.2691 | 13 scene x 5 seeds pooled mean。 |
| uniform_stride_ratio | 13 | 22.7165 | 0.6978 | 0.2583 | 13 scene mean。 |
| uniform - random seed mean | 13 scene pairs | +0.5211 | +0.0143 | -0.0108 | 对每个 scene 先求 random 5 seed mean，再与 uniform 比较；LPIPS 越低越好。 |

按 scene 对比：

| Scene | Train/Test | Random PSNR mean +/- sd | Random SSIM | Random LPIPS | Uniform PSNR | Uniform SSIM | Uniform LPIPS | Uniform-Random PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| db_drjohnson | 46/33 | 22.4274 +/- 0.5557 | 0.7048 | 0.2979 | 22.7489 | 0.7077 | 0.2936 | +0.3215 |
| db_playroom | 40/29 | 22.7298 +/- 0.6766 | 0.7784 | 0.2378 | 23.7406 | 0.7999 | 0.2236 | +1.0108 |
| mipnerf360_bicycle | 34/25 | 20.1522 +/- 0.7210 | 0.4808 | 0.3825 | 19.7752 | 0.4665 | 0.3889 | -0.3770 |
| mipnerf360_bonsai | 51/37 | 27.5462 +/- 0.4052 | 0.9000 | 0.1515 | 28.9883 | 0.9167 | 0.1356 | +1.4421 |
| mipnerf360_counter | 42/30 | 24.2247 +/- 0.6595 | 0.8143 | 0.1961 | 25.6509 | 0.8526 | 0.1710 | +1.4261 |
| mipnerf360_flowers | 31/22 | 16.1373 +/- 0.3878 | 0.3597 | 0.4356 | 16.1695 | 0.3690 | 0.4309 | +0.0322 |
| mipnerf360_garden | 33/24 | 23.1517 +/- 0.2060 | 0.7236 | 0.2449 | 23.8380 | 0.7421 | 0.2327 | +0.6863 |
| mipnerf360_kitchen | 49/35 | 27.1707 +/- 0.4665 | 0.8964 | 0.1179 | 28.1889 | 0.9137 | 0.1036 | +1.0182 |
| mipnerf360_room | 55/39 | 27.4050 +/- 0.5665 | 0.8873 | 0.1763 | 27.7214 | 0.8847 | 0.1728 | +0.3164 |
| mipnerf360_stump | 22/16 | 19.4950 +/- 0.2111 | 0.4110 | 0.4481 | 19.7489 | 0.4388 | 0.4361 | +0.2539 |
| mipnerf360_treehill | 25/18 | 17.3383 +/- 0.3027 | 0.3877 | 0.4697 | 17.4147 | 0.3907 | 0.4669 | +0.0764 |
| tandt_train | 53/38 | 18.0376 +/- 0.3080 | 0.6919 | 0.2319 | 18.2481 | 0.7249 | 0.2058 | +0.2105 |
| tandt_truck | 44/32 | 22.7249 +/- 0.2627 | 0.8504 | 0.1076 | 23.0813 | 0.8646 | 0.0964 | +0.3564 |

Uniform 相对 random seed mean：PSNR 在 12/13 个 scene 更高，SSIM 在 11/13 个 scene 更高，LPIPS 在 12/13 个 scene 更低。唯一 PSNR 下降的 scene 是 `mipnerf360_bicycle`。

## 相关性

- register cosine vs FastGS PSNR 的 Spearman rho：pending
- register cosine vs FastGS PSNR 的 Pearson r：pending
- 质量门：pending；random/uniform 的跨 scene / 多 seed FastGS 质量矩阵已完成，但还缺 register similarity 和对应相关性分析。

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
- random/uniform `images_4` 30k 矩阵稳定完成，当前 blocker 转为补齐 register/readout embedding、register similarity 和相关性分析。
