# 运行手册

## 当前约束

- 当前 `/` 约 `207G` 可用。
- `0007` full VGGT cache 占 `596G`，路径为 `caches/vggt_omega/0007_stage2_swap_gain_scaleup/swapgain1000_single8_images512`。
- 因此 0008 第一阶段禁止启动新的大规模 VGGT cache，只做 pose/manifest 离线诊断。

## 步骤 0: 空间与输入检查

```bash
df -h /home/m
du -sh caches/vggt_omega/0007_stage2_swap_gain_scaleup/swapgain1000_single8_images512 \
  caches/image_features/0007/swapgain1000_dinov2_vits14 \
  runs/0007_stage2_swap_gain_scaleup 2>/dev/null
test -f docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_manifest.json
```

预期：

- manifest 存在。
- 可用空间低于 `650G` 时，不启动阶段 C/D 的 VGGT 评估。

## 步骤 1: 实现角度阈值 subset builder

新增脚本建议：

```text
scripts/prepare_stage2_pose_angle_keyframe_subsets.py
```

输入：

```bash
--manifest docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_manifest.json
--out-json runs/0008_stage2_pose_angle_keyframing/pose_angle_subsets.json
--out-summary docs/experiments/0008_stage2_pose_angle_keyframing/keyframe_policy_summary.md
--candidate-tag 20
--thresholds-deg 10,15,20
```

实现要点：

- 从 `transform_matrix` 或 `pose_path` 读取 camera-to-world pose。
- 用 camera center 的 robust mean/median 估计 scene center。
- 计算 `azimuth`、`elevation` 和球面角距离。
- 生成固定 K=20 的 subsets。
- 记录每个方法的覆盖统计、overlap ratio、帧数异常。

输出方法名：

```text
pose_angle10_keyframe20
pose_angle15_keyframe20
pose_angle20_keyframe20
pose_farthest_angle20
pose_hybrid_uniform_angle20
uniform20
```

## 步骤 2: 离线诊断

运行：

```bash
PYTHONPATH=scripts:src python scripts/prepare_stage2_pose_angle_keyframe_subsets.py \
  --manifest docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_manifest.json \
  --out-json runs/0008_stage2_pose_angle_keyframing/pose_angle_subsets.json \
  --out-summary docs/experiments/0008_stage2_pose_angle_keyframing/keyframe_policy_summary.md \
  --candidate-tag 20 \
  --thresholds-deg 10,15,20
```

检查：

```bash
sed -n '1,160p' docs/experiments/0008_stage2_pose_angle_keyframing/keyframe_policy_summary.md
```

通过标准：

- 所有方法在大多数场景能稳定输出 20 帧。
- `pose_*` 相比 `uniform20` 有更高的方位/仰角覆盖。
- `pose_hybrid_uniform_angle20` 不应与 `uniform20` 完全重合，也不应过度集中在序列开头。

## 步骤 3: 复用 0007 标签做弱代理评估

新增或扩展脚本：

```text
scripts/evaluate_stage2_pose_angle_keyframes_against_swap_labels.py
```

目标：

- 对比 pose-angle subsets 与 `uniform20` 的 frame 差异。
- 如果差异可映射到 `0007` 的 single-swap labels，则读取该 swap 的 target gain。
- 输出 angle strategy 与 positive/negative swap 的命中率。

输入：

```text
runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8/augmented_hardlabel_train_labels.csv
runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8/augmented_cache_jobs.json
runs/0008_stage2_pose_angle_keyframing/pose_angle_subsets.json
```

产物：

```text
docs/experiments/0008_stage2_pose_angle_keyframing/proxy_against_0007_swap_labels.md
```

## 步骤 4: VGGT smoke

只有满足以下条件才启动：

- 阶段 2/3 显示角度策略不是纯噪声。
- `/` 可用空间至少 `300G`，更稳妥是 `650G`。
- 明确优先复用已有 full cache，避免重复生成 full outputs。

建议 smoke：

```text
50 scenes
25 DL3DV + 25 WildRGBD
methods:
  uniform20
  pose_angle10_keyframe20
  pose_farthest_angle20
  pose_hybrid_uniform_angle20
```

输出：

```text
runs/0008_stage2_pose_angle_keyframing/pose_angle_smoke50/
caches/vggt_omega/0008_stage2_pose_angle_keyframing/pose_angle_smoke50_images512/
```

停止条件：

- 可用空间低于 `150G`。
- 单场景 cache 成本明显高于 `0007`。
- 任何方法生成的 subset 帧数不是 20 且无法修复。

## 步骤 5: 正式评估

Smoke 通过后再做：

1. `300` scenes，与 `0006` 对齐。
2. 如 300 scenes 稳定正，再考虑 `1000` scenes。

正式评估前必须重新检查磁盘。如果仍只有约 `207G` 可用，应先决定是否删除 `0007` 的 `596G` VGGT cache。

## 结果整理

更新：

```text
docs/experiments/0008_stage2_pose_angle_keyframing/results.md
docs/experiments/0008_stage2_pose_angle_keyframing/review.md
```

必须记录：

- 每个方法的 coverage 统计。
- 相对 `uniform20` 的 VGGT-native delta。
- dataset-wise delta。
- cache size 与最终剩余磁盘。
- 是否值得把 pose-angle keyframing 作为固定 baseline。
