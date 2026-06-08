# Stage 2.0 Hard-Label Readout Manifest 摘要

- 创建日期: 2026-06-07
- 数据根目录: `/home/m/dataset/ltm_datasets`
- 入选 scenes: 100
- WildRGBD scenes 数量: 50
- DL3DV scenes 数量: 50
- Full frames: WildRGBD 100, DL3DV 80
- Hard subset jobs 数量: 1200
- VGGT cache jobs 总数: 1300
- Random seed: 20260607（固定随机种子）
- 排除的 validation scenes: 30

## 入选 Scene 统计

| Dataset | 数量 |
|---|---:|
| DL3DV-ALL-480P | 50 |
| wildrgbd_harrison | 50 |

## Subset 方法

- `random20_seed000` ... `random20_seed004`
- `random50_seed000` ... `random50_seed002`
- `uniform20`, `uniform50`
- `contiguous20_seed000`, `contiguous50_seed000`

## Label 目标

Hard labels 由每个 subset 的 VGGT-native depth/pose/point-map 输出与 full-view VGGT cache 中同一批图像的输出对比得到。
训练目标是按 scene 做 z-score 后的 `pose_rotation_mean_deg`、`pointmap_rmse_norm` 和 `depth_log_rmse` 求和。
