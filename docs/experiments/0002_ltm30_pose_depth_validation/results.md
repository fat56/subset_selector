# LTM30 Pose/Depth Validation Results

## Run

- Date: 2026-06-07
- Script: `scripts/run_ltm30_pose_depth_validation.py`
- Command:

```bash
/home/m/project/ltm/vggt-omega/.venv/bin/python \
  scripts/run_ltm30_pose_depth_validation.py \
  --random-seeds 5
```

- Scenes: 30
- Methods per scene: `full`, `random20_seed000` ... `random20_seed004`, `uniform20`
- VGGT cache jobs: 210/210 succeeded
- Output: `docs/experiments/0002_ltm30_pose_depth_validation/native_geometry/`

## Outputs

- `ltm30_register_similarity.csv`: register mean cosine to the scene `full` cache.
- `ltm30_subset_native_consistency.csv`: subset-vs-full VGGT-native depth, pose, and derived point-map consistency.
- `ltm30_gt_geometry_metrics.csv`: VGGT output against WildRGBD sensor depth and `camera_pose`.
- `ltm30_native_correlation_summary.csv`: scene-wise correlation summary for subset-vs-full metrics.
- `ltm30_gt_correlation_summary.csv`: scene-wise correlation summary for sensor GT metrics.
- `ltm30_native_method_summary.csv` and `ltm30_gt_method_summary.csv`: method-level averages.

## Key Results

Subset-vs-full VGGT-native consistency has a clear signal:

| Metric | Mean Spearman | Expected Sign | Best Match |
|---|---:|---:|---:|
| `pose_rotation_mean_deg` | -0.5429 | 29/30 | 21/30 |
| `pointmap_rmse_norm` | -0.5181 | 28/30 | 22/30 |
| `pose_rotation_median_deg` | -0.5067 | 29/30 | 22/30 |
| `depth_log_rmse` | -0.4990 | 28/30 | 25/30 |
| `depth_absrel_mean` | -0.4819 | 30/30 | 24/30 |
| `pose_center_rmse_norm` | -0.4857 | 26/30 | 19/30 |

Against WildRGBD sensor GT, depth has a weaker but still useful direction; pose is weak:

| Metric | Mean Spearman | Expected Sign | Best Match |
|---|---:|---:|---:|
| `gt_depth_absrel_mean` | -0.3512 | 25/30 | 8/30 |
| `gt_depth_absrel_median` | -0.3131 | 24/30 | 8/30 |
| `gt_pose_fov_abs_mean` | -0.2786 | 22/30 | 14/30 |
| `gt_pose_center_rmse_norm` | 0.0643 | 16/30 | 1/30 |
| `gt_pose_rotation_mean_deg` | 0.0571 | 12/30 | 1/30 |

Method averages also show a sane ordering for native consistency: `uniform20` has the highest register cosine (`0.9993`) and the lowest average subset-vs-full depth/pose/point-map errors. Among random seeds, higher register cosine generally corresponds to lower native geometry error, though not perfectly.

## Interpretation

This validates the earlier core judgment more strongly than the 13-scene Stage 1 run: mean-pooled register tokens are a meaningful proxy for VGGT's own 3D consistency. The signal is strongest for pose rotation, point-map RMSE, and depth consistency when the reference is the same scene's `full` VGGT output.

The external WildRGBD sensor-depth result is weaker but still directionally useful for depth. External pose GT is currently not reliable as a register-token quality target in this direct form. Likely reasons:

- VGGT pose is decoded in its own camera convention, while WildRGBD `camera_pose` convention is not fully normalized here.
- The current analysis uses one global similarity alignment per method and does not calibrate local camera axes beyond that.
- Register cosine measures subset agreement with VGGT's internal scene representation, not necessarily absolute metric-pose accuracy against an external dataset.

## Decision

Use LTM30 primarily as a stronger VGGT-native validation set for Stage 2 proxy design. For the first selector/readout training target, prioritize:

- subset-vs-full `pose_rotation_mean_deg`
- `pointmap_rmse_norm`
- `depth_log_rmse` / `depth_absrel_mean`

Use sensor `gt_depth_*` as a secondary sanity check. Do not use direct `gt_pose_*` as the main training or gate metric until the pose convention and axis calibration are audited.
