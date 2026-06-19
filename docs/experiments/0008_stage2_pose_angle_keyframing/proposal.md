# 0008 基于视角角度阈值的关键帧选择

## 元信息

- 实验 ID: `0008_stage2_pose_angle_keyframing`
- 阶段: `stage2`
- 状态: 计划中
- 创建时间: 2026-06-20
- 配置: `configs/experiments/0008_stage2_pose_angle_keyframing.yaml`
- 参考: [Marwan99/kv_tracker](https://github.com/Marwan99/kv_tracker), [KV-Tracker arXiv](https://arxiv.org/abs/2512.22581)

## 背景

`0007` 说明：继续堆 direct swap-gain labels 并不能让 image-only student 稳定超过 `uniform20`。问题不在 teacher headroom，而在 student 难以从纯图像特征可靠判断哪些 swap 真正有用。

KV-Tracker 提供了一个更朴素但有启发的方向：在线 pose tracking 中维护 keyframes，用相机视角差异保证 baseline 和视点多样性。论文/项目中的 keyframing 思路是：当当前帧与现有关键帧集合之间的最小方位角或仰角差异超过阈值时，将当前帧加入关键帧集合。对象级实验使用过 `10°` 的角度阈值。

这个策略和我们的 fixed-K subset selection 有天然对应关系：如果一个 subset 的目标是覆盖三维结构，显式保证视角分散可能比从 image-only embedding 学一个隐式 swap rule 更稳。

## 核心问题

基于已知或在线预测 pose 的角度阈值 keyframing，是否能在 VGGT-native geometry target 上稳定优于 `uniform20`，或者至少成为一个比当前 learned image-only selector 更可靠的非学习 baseline？

## 假设

角度阈值 keyframing 会在以下场景更有优势：

- 相机运动明显、视角覆盖不足时，`uniform20` 可能采到时间上均匀但角度冗余的帧。
- 单纯 image-only feature 难以判断 baseline 是否足够，而 pose angular coverage 可以直接表达这一点。
- 如果阈值选择足够保守，keyframing 不需要学习阈值，因此应比 `0007` 的 val-selected student 更不容易 over-swap。

主要风险也很清楚：

- WildRGBD / DL3DV 的 pose metadata 是离线真值或 COLMAP 结果，不等同于 KV-Tracker 的在线 pose prediction；第一版验证的是“角度策略是否值得”，不是“在线系统完整可用”。
- `uniform20` 在很多历史实验中非常强，单纯角度覆盖可能牺牲纹理质量、尺度分布或时间连续性。
- 当前磁盘只剩约 `207G`，不能直接启动新的大规模 VGGT cache。

## 方法概述

### 角度定义

对每个 frame 的 camera-to-world transform 取相机中心 `c`。用场景中心 `o` 作为参考点，优先取所有 camera centers 的均值；如果该均值不稳定，再尝试用 full point/depth bounds 或 robust median。

定义从场景中心指向相机的方向：

```text
v_t = normalize(c_t - o)
azimuth phi_t   = atan2(v_t.x, v_t.z)
elevation theta_t = asin(v_t.y)
```

为了避免角度 wrap-around 错误，方位角差使用 circular distance：

```text
d_phi(a, b) = abs(((a - b + pi) mod 2pi) - pi)
```

核心 keyframe 规则：

```text
frame t is keyframe if
  min_kf d_phi(phi_t, phi_kf) > tau_phi
  or
  min_kf abs(theta_t - theta_kf) > tau_theta
```

第一帧总是 keyframe。若 keyframes 超过 `K=20`，用 farthest coverage / temporal thinning 压回 20；若少于 20，则用最大角度空洞补帧，直到达到固定 K。

### 待比较策略

第一阶段只生成 subsets，不运行新 VGGT：

| 方法 | 说明 |
|---|---|
| `uniform20` | 现有强 baseline。 |
| `pose_angle10_keyframe20` | KV-Tracker 风格，`tau_phi=tau_theta=10°`。 |
| `pose_angle15_keyframe20` | 更保守，减少 keyframe 数量。 |
| `pose_angle20_keyframe20` | 更强 baseline 间隔，测试视角覆盖上限。 |
| `pose_farthest_angle20` | 不用阈值，直接贪心最大化到已有集合的最小球面角距离。 |
| `pose_hybrid_uniform_angle20` | 先取少量 uniform anchors，再用角度 farthest 补足，降低时间偏置。 |

第二阶段如果第一阶段指标有希望，再对上述 subset 运行 VGGT-native evaluation。

## 实验阶段

### 阶段 A: 零 VGGT 的离线诊断

输入：

- `docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_manifest.json`
- 其中每个 frame 已带 `transform_matrix` / `pose_path`。

输出：

- `docs/experiments/0008_stage2_pose_angle_keyframing/keyframe_policy_summary.md`
- `runs/0008_stage2_pose_angle_keyframing/pose_angle_subsets.json`

诊断指标：

- 每个方法实际选中帧数是否稳定等于 `20`。
- 方位角覆盖范围、仰角覆盖范围。
- 选中帧对之间的最小/平均球面角距离。
- 与 `uniform20` 的 overlap ratio。
- 与 `0007` 中 best single-swap added/removed frames 的重合度，用来判断角度策略是否覆盖 teacher 认为有价值的方向。

这一阶段几乎不占磁盘，是当前空间状态下必须先做的验证。

### 阶段 B: 复用已有 0007 标签做代理评估

不生成新 VGGT cache，先用 `0007` 已有标签做弱代理：

- 如果 angle keyframe subset 与 `uniform20` 只差 1 个 frame，且该替换在 `0007` 的 `swapgain20_dino1_rank*` labels 中出现，则直接读取对应 target error。
- 统计 angle strategy 是否倾向于命中正 gain swap，是否避开负 gain swap。

这不能替代正式评估，但可以判断角度策略和现有 teacher signal 是否同向。

### 阶段 C: 小规模 VGGT smoke

仅当阶段 A/B 有希望且磁盘空间足够时启动。

目标：

- `50` scenes，DL3DV/WildRGBD 各 `25`。
- 评估 `uniform20`、`pose_angle10_keyframe20`、`pose_farthest_angle20`、`pose_hybrid_uniform_angle20`。
- 优先复用已有 full cache；不要生成 1000 场景新 cache。

通过条件：

- `pose_*` 至少一个方法相对 `uniform20` 的 mean delta `>= +0.05`。
- 至少 `55%` scenes 不差于 `uniform20`。
- 两个数据集都不能明显为负。

### 阶段 D: 300/1000 场景正式评估

只有 smoke 通过后再做。

优先路线：

1. 先跑 `300` scenes，与 `0006` 规模可比。
2. 若稳定正，再考虑 `1000` scenes。
3. 当前磁盘不允许直接开大规模 VGGT；启动前需要释放 0007 full VGGT cache 或确认有至少 `650G` 可用空间。

## 评价指标

主指标：

```text
uniform_minus_pose_keyframe_error
```

即 `uniform20` 的 target error 减去 pose-keyframe subset 的 target error，正数表示 pose-keyframe 更好。

辅助指标：

- win rate vs `uniform20`。
- dataset-wise delta: DL3DV / WildRGBD。
- overlap ratio vs `uniform20`。
- angle coverage gain vs `uniform20`。
- VGGT-native primary metrics: `pose_rotation_mean_deg`、`pointmap_rmse_norm`、`depth_log_rmse`。
- cache size / scene，避免再次低估磁盘。

## 晋级标准

阶段 C smoke 晋级到正式评估：

- Mean delta `>= +0.05`。
- Win rate vs `uniform20` `>= 0.55`。
- DL3DV 和 WildRGBD 的 delta 都 `>= -0.02`。

正式评估晋级为 baseline：

- 300 scenes 上 mean delta `>= +0.05`。
- Median delta `> 0`。
- Worst dataset delta `>= -0.02`。
- 不依赖 learned threshold 或 test-oracle scan。

如果 300 scenes 结果正但不强，则作为 diagnostic baseline 保留，不作为 selector 主线。

## 与 0007 的关系

`0007` 失败说明：从 image-only feature 直接学习 single-swap gain，很容易在 validation threshold 上过拟合。`0008` 换一个角度：不再先学“哪一帧看起来好”，而是显式把 pose baseline 和视角多样性作为选择规则。

如果 `0008` 成功，后续可以做两件事：

- 把 pose-angle keyframe 作为非学习 baseline，和 learned selector 长期对照。
- 训练 student 去模仿 angle/pose coverage，而不是直接回归 noisy VGGT gain。

如果 `0008` 失败，说明我们的目标函数并不简单偏好角度覆盖，下一步应转向更强的几何/可见性特征，而不是继续只调阈值。
