# 复盘

## 决策

阶段 A/B 完成，但暂不启动 VGGT smoke。

理由：

- Pose-angle keyframing 明显改善了视角覆盖和最小视角间距。
- 但代理评估显示，它与 `0007` single-swap teacher 信号不是强同向。
- 这些方法通常一次替换 `10-14` 帧，不是 single-swap，因此不能用 `0007` labels 直接证明它会降低 VGGT-native target error。
- 当前磁盘只有约 `207G` 可用，低于 0008 VGGT smoke 的最低空间要求。

## 问题回答

| 问题 | 当前回答 |
|---|---|
| 角度阈值 keyframing 是否显著提升视角覆盖？ | 是。Azimuth coverage 从 `281.20°` 提高到最高 `307.36°`；min pairwise angle 从 `3.08°` 提高到最高 `14.63°`。 |
| 角度策略选中的帧是否与 `0007` positive swap-gain 信号同向？ | 弱同向但不强。Added frame 命中 positive swap 的比例只有约 `5%-6%`，命中 best added frame 的场景比例约 `19%-24%`。 |
| 小规模 VGGT smoke 是否优于 `uniform20`？ | 尚未运行。当前磁盘不足，且代理信号不够强，不建议贸然启动。 |
| 该策略是否在 DL3DV 和 WildRGBD 上都不明显变差？ | 覆盖指标两边都有改善；VGGT-native target 尚未验证。 |
| 是否值得把它作为长期 baseline？ | 值得保留为候选 baseline，但需要 VGGT smoke 或更局部的 pose-angle variant 才能定论。 |

## 关键观察

- `uniform20` 的平均选帧数是 `18`，因为 DL3DV 是 `80 x 20% = 16` 帧，WildRGBD 是 `100 x 20% = 20` 帧。
- `pose_angle20_keyframe20` 的 azimuth coverage 最强，且 best single-swap added-frame hit rate 最高，达到 `0.244`。
- `pose_farthest_angle20` 的 min pairwise angle 最高，达到 `14.63°`。
- `pose_hybrid_uniform_angle20` 最保守，overlap with uniform 为 `0.406`，但 best-hit rate 较低。
- 当前 pose-angle variants 与 `uniform20` 差异太大，代理评估缺少直接可映射的 single-swap case。

## 风险

- 第一版使用 metadata/离线 pose，不等同于在线 pose predictor。
- 角度覆盖不一定等价于 VGGT-native target 变好。
- 当前磁盘空间不足以直接做大规模新 cache。
- 如果直接跑 full-subset pose-angle VGGT smoke，可能会花掉较多空间，却发现收益并不稳定。

## 下一步建议

1. 先做 `pose_angle_local_swap`：从 `uniform20` 出发，只替换 1、2、4 个角度最冗余的 uniform frames。
2. 对 local variants 复用 `0007` single-swap labels，这样可以得到更强的代理判断。
3. 如果 local proxy 明显正，再考虑 50-scene VGGT smoke。
4. 若要直接跑当前 full pose-angle smoke，需要先释放空间；候选是删除已完成标签后的 `0007` full VGGT cache。
