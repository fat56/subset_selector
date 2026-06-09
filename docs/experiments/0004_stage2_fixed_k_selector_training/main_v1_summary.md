# Stage 2 固定 K Selector Main V1 Manifest 摘要

- 创建日期: 2026-06-08
- 入选 scenes: 2138
- 总 frames: 105204
- Cache 策略: cache-light `selector_features.pt`；不保存 `depth`、`depth_conf` 或 full dense VGGT outputs。
- ScanNet 来源: 使用 `data/raw/ltm_datasets/yifei_scannetv2_hf`。

## 可用 Scene 统计

| 数据集 | 可用 scenes | 入选 scenes |
|---|---:|---:|
| bonn | 26 | 26 |
| bridgedata_v2 | 25446 | 1000 |
| nyuv2 | 549 | 549 |
| tartanair | 163 | 163 |
| yifei_scannetv2_hf | 1510 | 400 |

## Split

| Split | Scenes |
|---|---:|
| test | 205 |
| train | 1728 |
| val | 205 |
