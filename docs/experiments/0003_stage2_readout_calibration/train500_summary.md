# Stage 2.0 Readout Train500 Manifest 摘要

- 创建日期: 2026-06-07
- 数据根目录: `/home/m/dataset/ltm_datasets`
- 入选 scenes: 500
- 每个 scene 的 full frames: 16
- Full frames 总数: 8000
- Random seed: 20260607（固定随机种子）
- 排除的 validation scenes: 30

## 入选 Scene 统计

| Key | 数量 |
|---|---:|
| dataset:DL3DV-ALL-480P | 250 |
| dataset:wildrgbd_harrison | 250 |
| depth:colmap_photometric_bin | 250 |
| depth:sensor_depth_png | 250 |

## 训练范围

- 这个 manifest 只缓存每个 scene 的 full-view token set。
- 训练时从 cached full tokens 在线采样 subset masks。
- LTM30 hard subset native metrics 在这个 MVP 中只用于 validation。
