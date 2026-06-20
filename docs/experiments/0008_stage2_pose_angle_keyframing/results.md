# 结果

## 当前状态

阶段 A/B 已完成：

- 已实现 `scripts/prepare_stage2_pose_angle_keyframe_subsets.py`。
- 已实现 `scripts/evaluate_stage2_pose_angle_keyframes_against_swap_labels.py`。
- 已在 `0007` 的 1000-scene manifest 上完成 pose-angle keyframe 离线诊断。
- 已复用 `0007` single-swap labels 完成弱代理评估。
- 未启动 VGGT smoke，因为当前 `/` 只有约 `207G` 可用，低于计划中的 `300G` 最低线和 `650G` 推荐线。

## 运行记录

| Run ID | 场景数 | 方法 | 是否需要新 VGGT cache | 状态 |
|---|---:|---|---|---|
| `pose_angle_offline_1000` | 1000 | angle coverage diagnostics | 否 | 完成 |
| `pose_angle_proxy_against_0007` | 1000 | 与 `0007` single-swap labels 对齐 | 否 | 完成 |
| `pose_angle_smoke50` | 50 | `uniform20` + 3 个 pose-angle methods | 是，小规模 | 暂停，磁盘不足 |
| `pose_angle_eval300` | 300 | smoke 通过后确定 | 是 | 暂不启动 |

产物：

- `docs/experiments/0008_stage2_pose_angle_keyframing/keyframe_policy_summary.md`
- `docs/experiments/0008_stage2_pose_angle_keyframing/proxy_against_0007_swap_labels.md`
- `runs/0008_stage2_pose_angle_keyframing/pose_angle_subsets.json`
- `runs/0008_stage2_pose_angle_keyframing/proxy_against_0007_swap_labels.json`

## 离线覆盖诊断

`uniform20` 的平均选帧数是 `18`，这是因为 DL3DV full count 为 80 时 `20%` 预算是 16 帧，WildRGBD full count 为 100 时是 20 帧。

| 方法 | Mean azimuth coverage | Mean elevation coverage | Mean pairwise angle | Min pairwise angle | Overlap vs `uniform20` |
|---|---:|---:|---:|---:|---:|
| `uniform20` | `281.20` | `91.97` | `93.60` | `3.08` | `1.000` |
| `pose_angle10_keyframe20` | `297.93` | `95.33` | `92.02` | `11.95` | `0.235` |
| `pose_angle15_keyframe20` | `301.90` | `95.72` | `92.14` | `12.02` | `0.237` |
| `pose_angle20_keyframe20` | `307.36` | `95.82` | `92.32` | `9.88` | `0.235` |
| `pose_farthest_angle20` | `294.63` | `93.95` | `93.19` | `14.63` | `0.240` |
| `pose_hybrid_uniform_angle20` | `292.77` | `93.87` | `93.15` | `9.77` | `0.406` |

解读：

- Pose-angle 方法确实提升了 azimuth/elevation coverage。
- 最明显的改善是最小 pairwise angle：`uniform20` 只有 `3.08°`，pose-angle 方法提高到约 `9.8°-14.6°`。
- `pose_angle20_keyframe20` 的 azimuth coverage 最大，达到 `307.36°`。
- `pose_farthest_angle20` 的最小 pairwise angle 最大，达到 `14.63°`。
- `pose_hybrid_uniform_angle20` 保留更多 uniform anchor，overlap 为 `0.406`，是最保守的 angle variant。

Dataset-wise 摘要：

| 方法 | DL3DV azimuth | DL3DV elevation | WildRGBD azimuth | WildRGBD elevation |
|---|---:|---:|---:|---:|
| `uniform20` | `237.07` | `143.66` | `325.32` | `40.28` |
| `pose_angle10_keyframe20` | `264.51` | `150.69` | `331.35` | `39.98` |
| `pose_angle15_keyframe20` | `273.21` | `151.43` | `330.58` | `40.01` |
| `pose_angle20_keyframe20` | `280.24` | `151.52` | `334.47` | `40.13` |
| `pose_farthest_angle20` | `255.19` | `147.77` | `334.07` | `40.13` |
| `pose_hybrid_uniform_angle20` | `255.35` | `147.63` | `330.19` | `40.11` |

## 代理评估

复用 `0007` single-swap labels 时，一个重要限制是：pose-angle subsets 通常不是 single-swap，而是一次替换约 10 到 14 帧。因此 `0007` 的 single-swap labels 不能直接给这些完整 subsets 打分，只能作为“新增帧是否碰到 positive/best single-swap added frame”的弱代理。

| 方法 | Added 数 | Overlap vs uniform | Single-swap diff | 可映射率 | Added 命中正 swap | Added 命中 best swap |
|---|---:|---:|---:|---:|---:|---:|
| `pose_angle10_keyframe20` | `13.79` | `0.235` | `0.000` | `0.000` | `0.052` | `0.214` |
| `pose_angle15_keyframe20` | `13.74` | `0.237` | `0.000` | `0.000` | `0.054` | `0.227` |
| `pose_angle20_keyframe20` | `13.79` | `0.235` | `0.000` | `0.000` | `0.053` | `0.244` |
| `pose_farthest_angle20` | `13.69` | `0.240` | `0.000` | `0.000` | `0.058` | `0.232` |
| `pose_hybrid_uniform_angle20` | `10.74` | `0.406` | `0.000` | `0.000` | `0.061` | `0.191` |

解读：

- `0007` labels 无法直接映射完整 pose-angle subset；可映射率为 `0` 是预期结果，不代表策略无效。
- Angle variants 与 `uniform20` 差异很大，平均 added frames 为 `10.74-13.79`。
- 命中 best single-swap added frame 的场景比例约 `19.1%-24.4%`，其中 `pose_angle20_keyframe20` 最高。
- 但 added frames 命中 positive swap 的逐帧比例只有约 `5%-6%`，说明 angle coverage 与 `0007` 的 DINO single-swap teacher 信号并不强同向。

## VGGT-native 结果

尚未运行。

当前不启动 VGGT smoke 的原因：

- `/` 可用空间约 `207G`。
- `0007` full VGGT cache 仍占 `596G`。
- 0008 计划中 VGGT smoke 最低要求为 `300G`，推荐 `650G`。
- 离线/代理结果只能说明 angle 方法提高视角覆盖，尚不足以在空间紧张时直接开新 cache。

## 空间记录

- 运行前 `/` 可用空间约 `207G`。
- 离线阶段完成后 `/` 仍约 `207G`。
- `runs/0008_stage2_pose_angle_keyframing` 约 `49M`。
- 未新增 VGGT cache。

## 初步结论

`0008` 阶段 A/B 证明了角度阈值 keyframing 在几何覆盖指标上确实有效，尤其能显著提高最小视角间距；但它与 `0007` single-swap teacher 的弱代理信号不强一致，而且通常一次替换很多帧，不能用 single-swap labels 直接判断 VGGT-native 收益。

因此当前结论是：保留该路线，但暂不进入 VGGT smoke。下一步更稳的做法是先设计更接近 `uniform20` 的 pose-angle local variants，例如只替换 1-4 个角度冗余帧，再用已有 0007 single-swap labels 做更可解释的代理评估；或者在清理 0007 大 cache 后，直接跑 50-scene VGGT smoke。
