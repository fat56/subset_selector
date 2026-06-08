# LTM30 Pose/Depth 验证结果

## 运行信息

- 日期: 2026-06-07
- 脚本: `scripts/run_ltm30_pose_depth_validation.py`
- 命令:

```bash
/home/m/project/ltm/vggt-omega/.venv/bin/python \
  scripts/run_ltm30_pose_depth_validation.py \
  --random-seeds 5
```

- Scene 数: 30
- 每个 scene 的方法: `full`, `random20_seed000` ... `random20_seed004`, `uniform20`
- VGGT cache jobs: 210/210 成功
- 输出目录: `docs/experiments/0002_ltm30_pose_depth_validation/native_geometry/`

## 输出文件

- `ltm30_register_similarity.csv`: subset 与同 scene `full` cache 的 register mean cosine。
- `ltm30_subset_native_consistency.csv`: subset-vs-full 的 VGGT-native depth、pose 和派生 point-map consistency。
- `ltm30_gt_geometry_metrics.csv`: VGGT 输出与 WildRGBD sensor depth / `camera_pose` 的对比。
- `ltm30_native_correlation_summary.csv`: subset-vs-full 指标的 scene-wise correlation 汇总。
- `ltm30_gt_correlation_summary.csv`: sensor GT 指标的 scene-wise correlation 汇总。
- `ltm30_native_method_summary.csv` 和 `ltm30_gt_method_summary.csv`: method-level 平均值。

## 关键结果

subset-vs-full 的 VGGT-native consistency 有明确方向性：

| 指标 | Mean Spearman | 期望符号 | Best Match |
|---|---:|---:|---:|
| `pose_rotation_mean_deg` | -0.5429 | 29/30 | 21/30 |
| `pointmap_rmse_norm` | -0.5181 | 28/30 | 22/30 |
| `pose_rotation_median_deg` | -0.5067 | 29/30 | 22/30 |
| `depth_log_rmse` | -0.4990 | 28/30 | 25/30 |
| `depth_absrel_mean` | -0.4819 | 30/30 | 24/30 |
| `pose_center_rmse_norm` | -0.4857 | 26/30 | 19/30 |

对 WildRGBD sensor GT 来看，depth 的方向性较弱但仍可用；pose 信号较弱：

| 指标 | Mean Spearman | 期望符号 | Best Match |
|---|---:|---:|---:|
| `gt_depth_absrel_mean` | -0.3512 | 25/30 | 8/30 |
| `gt_depth_absrel_median` | -0.3131 | 24/30 | 8/30 |
| `gt_pose_fov_abs_mean` | -0.2786 | 22/30 | 14/30 |
| `gt_pose_center_rmse_norm` | 0.0643 | 16/30 | 1/30 |
| `gt_pose_rotation_mean_deg` | 0.0571 | 12/30 | 1/30 |

method-level 平均值也符合 native consistency 的直觉排序：`uniform20` 的 register cosine 最高（`0.9993`），平均 subset-vs-full depth/pose/point-map error 最低。在多个 random seed 之间，更高的 register cosine 通常对应更低的 native geometry error，虽然不是完全单调。

## 解读

这比 13-scene 的 Stage 1 run 更强地验证了之前的核心判断：mean-pooled register tokens 可以作为 VGGT 自身 3D consistency 的有效 proxy。当参考是同一 scene 的 `full` VGGT 输出时，pose rotation、point-map RMSE 和 depth consistency 的信号最强。

外部 WildRGBD sensor-depth 结果较弱，但对 depth 仍有方向性参考价值。external pose GT 目前不适合直接作为 register-token quality target。可能原因：

- VGGT pose 使用自己的 camera convention 解码，而这里还没有完全规范化 WildRGBD `camera_pose` convention。
- 当前分析对每个 method 只做一次 global similarity alignment，没有进一步校准局部 camera axes。
- Register cosine 衡量的是 subset 与 VGGT 内部 scene representation 的一致性，不一定等价于对外部数据集的绝对 metric-pose accuracy。

## 决策

LTM30 主要作为 Stage 2 proxy design 的强 VGGT-native 验证集。第一版 selector/readout 训练目标优先使用：

- subset-vs-full `pose_rotation_mean_deg`
- `pointmap_rmse_norm`
- `depth_log_rmse` / `depth_absrel_mean`

sensor `gt_depth_*` 只作为 secondary sanity check。在 pose convention 和 axis calibration 审计完成前，不把 direct `gt_pose_*` 作为主训练目标或 gate metric。
