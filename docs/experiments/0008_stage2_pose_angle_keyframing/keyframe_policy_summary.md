# 0008 Pose-Angle Keyframe 离线诊断

- Manifest: `/home/m/project/ltm/selector/docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_manifest.json`
- 场景数: `1000`
- 数据集分布: `{"DL3DV-ALL-480P": 500, "wildrgbd_harrison": 500}`
- 阈值: `[10.0, 15.0, 20.0]` degrees

## 方法汇总

| 方法 | 选帧数 | Azimuth 覆盖 | Elevation 覆盖 | Mean pair angle | Min pair angle | Overlap vs uniform | Temporal span |
|---|---:|---:|---:|---:|---:|---:|---:|
| `uniform20` | 18.00 | 281.20 | 91.97 | 93.60 | 3.08 | 1.000 | 89.00 |
| `pose_angle10_keyframe20` | 18.00 | 297.93 | 95.33 | 92.02 | 11.95 | 0.235 | 75.28 |
| `pose_angle15_keyframe20` | 18.00 | 301.90 | 95.72 | 92.14 | 12.02 | 0.237 | 74.87 |
| `pose_angle20_keyframe20` | 18.00 | 307.36 | 95.82 | 92.32 | 9.88 | 0.235 | 78.76 |
| `pose_farthest_angle20` | 18.00 | 294.63 | 93.95 | 93.19 | 14.63 | 0.240 | 82.62 |
| `pose_hybrid_uniform_angle20` | 18.00 | 292.77 | 93.87 | 93.15 | 9.77 | 0.406 | 80.64 |

## Dataset-wise

### `uniform20`

| 数据集 | Azimuth 覆盖 | Elevation 覆盖 | Mean pair angle | Overlap vs uniform |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 237.07 | 143.66 | 93.34 | 1.000 |
| `wildrgbd_harrison` | 325.32 | 40.28 | 93.87 | 1.000 |

### `pose_angle10_keyframe20`

| 数据集 | Azimuth 覆盖 | Elevation 覆盖 | Mean pair angle | Overlap vs uniform |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 264.51 | 150.69 | 90.08 | 0.238 |
| `wildrgbd_harrison` | 331.35 | 39.98 | 93.96 | 0.231 |

### `pose_angle15_keyframe20`

| 数据集 | Azimuth 覆盖 | Elevation 覆盖 | Mean pair angle | Overlap vs uniform |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 273.21 | 151.43 | 90.01 | 0.238 |
| `wildrgbd_harrison` | 330.58 | 40.01 | 94.28 | 0.235 |

### `pose_angle20_keyframe20`

| 数据集 | Azimuth 覆盖 | Elevation 覆盖 | Mean pair angle | Overlap vs uniform |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 280.24 | 151.52 | 90.66 | 0.239 |
| `wildrgbd_harrison` | 334.47 | 40.13 | 93.98 | 0.230 |

### `pose_farthest_angle20`

| 数据集 | Azimuth 覆盖 | Elevation 覆盖 | Mean pair angle | Overlap vs uniform |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 255.19 | 147.77 | 92.40 | 0.247 |
| `wildrgbd_harrison` | 334.07 | 40.13 | 93.98 | 0.233 |

### `pose_hybrid_uniform_angle20`

| 数据集 | Azimuth 覆盖 | Elevation 覆盖 | Mean pair angle | Overlap vs uniform |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 255.35 | 147.63 | 92.18 | 0.426 |
| `wildrgbd_harrison` | 330.19 | 40.11 | 94.11 | 0.386 |

