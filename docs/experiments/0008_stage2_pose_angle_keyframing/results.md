# 结果

## 当前状态

尚未运行。本文档用于记录 `0008` 基于视角角度阈值的 keyframe selection 结果。

## 计划运行

| Run ID | 场景数 | 方法 | 是否需要新 VGGT cache | 状态 |
|---|---:|---|---|---|
| `pose_angle_offline_1000` | 1000 | angle coverage diagnostics | 否 | 计划中 |
| `pose_angle_proxy_against_0007` | 1000 | 与 `0007` single-swap labels 对齐 | 否 | 计划中 |
| `pose_angle_smoke50` | 50 | `uniform20` + 3 个 pose-angle methods | 是，小规模 | 等待前两步结果 |
| `pose_angle_eval300` | 300 | smoke 通过后确定 | 是 | 暂不启动 |

## 离线覆盖诊断

待填写：

| 方法 | Mean azimuth coverage | Mean elevation coverage | Mean pairwise angle | Overlap vs `uniform20` | 异常场景 |
|---|---:|---:|---:|---:|---:|
| `uniform20` | pending | pending | pending | `1.000` | pending |
| `pose_angle10_keyframe20` | pending | pending | pending | pending | pending |
| `pose_angle15_keyframe20` | pending | pending | pending | pending | pending |
| `pose_angle20_keyframe20` | pending | pending | pending | pending | pending |
| `pose_farthest_angle20` | pending | pending | pending | pending | pending |
| `pose_hybrid_uniform_angle20` | pending | pending | pending | pending | pending |

## 代理评估

待填写：复用 `0007` single-swap labels 时，pose-angle methods 命中 positive gain swap 的比例。

## VGGT-native 结果

待填写：

| 方法 | 场景数 | Mean delta vs `uniform20` | Median delta | Win rate | DL3DV delta | WildRGBD delta |
|---|---:|---:|---:|---:|---:|---:|
| `pose_angle10_keyframe20` | pending | pending | pending | pending | pending | pending |
| `pose_farthest_angle20` | pending | pending | pending | pending | pending | pending |
| `pose_hybrid_uniform_angle20` | pending | pending | pending | pending | pending | pending |

## 空间记录

初始状态：

- `/` 可用空间约 `207G`。
- `0007` full VGGT cache: `596G`。
- 因此第一阶段只做离线诊断，不启动新的大规模 VGGT cache。

## 初步结论

待运行后填写。
