# 0008 与 0007 Single-Swap Labels 的代理评估

- Subsets: `/home/m/project/ltm/selector/runs/0008_stage2_pose_angle_keyframing/pose_angle_subsets.json`
- Labels: `/home/m/project/ltm/selector/runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8/augmented_hardlabel_train_labels.csv`
- 场景数: `1000`
- 数据集分布: `{"DL3DV-ALL-480P": 500, "wildrgbd_harrison": 500}`

## 方法汇总

| 方法 | Added 数 | Overlap vs uniform | Single-swap diff | 可映射率 | 映射 gain mean | 映射正 gain | Added 命中正 swap | Added 命中 best swap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pose_angle10_keyframe20` | 13.79 | 0.235 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.052 | 0.214 |
| `pose_angle15_keyframe20` | 13.74 | 0.237 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.054 | 0.227 |
| `pose_angle20_keyframe20` | 13.79 | 0.235 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.053 | 0.244 |
| `pose_farthest_angle20` | 13.69 | 0.240 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.058 | 0.232 |
| `pose_hybrid_uniform_angle20` | 10.74 | 0.406 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.061 | 0.191 |

## Dataset-wise

### `pose_angle10_keyframe20`

| 数据集 | 可映射率 | 映射 gain mean | Added 命中正 swap | Added 命中 best swap |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 0.000 | 0.0000 | 0.057 | 0.190 |
| `wildrgbd_harrison` | 0.000 | 0.0000 | 0.048 | 0.238 |

### `pose_angle15_keyframe20`

| 数据集 | 可映射率 | 映射 gain mean | Added 命中正 swap | Added 命中 best swap |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 0.000 | 0.0000 | 0.057 | 0.176 |
| `wildrgbd_harrison` | 0.000 | 0.0000 | 0.052 | 0.278 |

### `pose_angle20_keyframe20`

| 数据集 | 可映射率 | 映射 gain mean | Added 命中正 swap | Added 命中 best swap |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 0.000 | 0.0000 | 0.055 | 0.204 |
| `wildrgbd_harrison` | 0.000 | 0.0000 | 0.052 | 0.284 |

### `pose_farthest_angle20`

| 数据集 | 可映射率 | 映射 gain mean | Added 命中正 swap | Added 命中 best swap |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 0.000 | 0.0000 | 0.066 | 0.224 |
| `wildrgbd_harrison` | 0.000 | 0.0000 | 0.050 | 0.240 |

### `pose_hybrid_uniform_angle20`

| 数据集 | 可映射率 | 映射 gain mean | Added 命中正 swap | Added 命中 best swap |
|---|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | 0.000 | 0.0000 | 0.069 | 0.176 |
| `wildrgbd_harrison` | 0.000 | 0.0000 | 0.054 | 0.206 |

