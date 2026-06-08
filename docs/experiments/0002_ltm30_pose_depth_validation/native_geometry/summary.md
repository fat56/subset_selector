# LTM30 Native Geometry 验证

- Scenes 数量: 30
- Cache jobs 数量: 210
- Random seeds 数量: 5
- GT rows 数量: 210
- Native subset rows 数量: 180
- 图像分辨率: 512 (balanced)

## GT 相关性汇总

| 指标 | Scenes | Mean Spearman | 符号 | Best Match |
|---|---:|---:|---:|---:|
| `gt_depth_absrel_mean` | 30 | -0.35119 | 25/30 | 8/30 |
| `gt_depth_absrel_median` | 30 | -0.313095 | 24/30 | 8/30 |
| `gt_depth_log_rmse` | 30 | -0.241667 | 22/30 | 1/30 |
| `gt_pose_center_median_norm` | 30 | 0.077381 | 14/30 | 1/30 |
| `gt_pose_center_p90_norm` | 30 | 0.091667 | 13/30 | 3/30 |
| `gt_pose_center_rmse_norm` | 30 | 0.064286 | 16/30 | 1/30 |
| `gt_pose_fov_abs_mean` | 30 | -0.278571 | 22/30 | 14/30 |
| `gt_pose_rotation_mean_deg` | 30 | 0.057143 | 12/30 | 1/30 |
| `gt_pose_rotation_median_deg` | 30 | 0.005952 | 13/30 | 2/30 |

## Native Subset 相关性汇总

| 指标 | Scenes | Mean Spearman | 符号 | Best Match |
|---|---:|---:|---:|---:|
| `depth_absrel_mean` | 30 | -0.481905 | 30/30 | 24/30 |
| `depth_absrel_median` | 30 | -0.398095 | 27/30 | 23/30 |
| `depth_log_rmse` | 30 | -0.499048 | 28/30 | 25/30 |
| `pointmap_l1_norm` | 30 | -0.481905 | 28/30 | 21/30 |
| `pointmap_median_norm` | 30 | -0.367619 | 26/30 | 18/30 |
| `pointmap_p90_norm` | 30 | -0.422857 | 24/30 | 18/30 |
| `pointmap_rmse_norm` | 30 | -0.518095 | 28/30 | 22/30 |
| `pose_center_median_norm` | 30 | -0.445714 | 25/30 | 16/30 |
| `pose_center_p90_norm` | 30 | -0.453333 | 25/30 | 16/30 |
| `pose_center_rmse_norm` | 30 | -0.485714 | 26/30 | 19/30 |
| `pose_fov_abs_mean` | 30 | -0.203809 | 22/30 | 11/30 |
| `pose_rotation_mean_deg` | 30 | -0.542857 | 29/30 | 21/30 |
| `pose_rotation_median_deg` | 30 | -0.506667 | 29/30 | 22/30 |
