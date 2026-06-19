# 复盘

## 决策

待定。`0008` 的第一步是低成本验证 pose-angle keyframing 是否值得进入 VGGT-native smoke。

## 需要回答的问题

| 问题 | 当前状态 |
|---|---|
| 角度阈值 keyframing 是否显著提升视角覆盖？ | 待离线诊断。 |
| 角度策略选中的帧是否与 `0007` positive swap-gain 信号同向？ | 待代理评估。 |
| 小规模 VGGT smoke 是否优于 `uniform20`？ | 待前两步通过后运行。 |
| 该策略是否在 DL3DV 和 WildRGBD 上都不明显变差？ | 待 VGGT-native 评估。 |
| 是否值得把它作为长期 baseline？ | 待定。 |

## 风险

- 该方法依赖 pose。第一版使用 metadata/离线 pose，不等同于在线 pose predictor。
- 角度覆盖不一定等价于 VGGT-native target 变好。
- 当前磁盘空间不足以直接做大规模新 cache。
- 如果场景中心估计不稳，azimuth/elevation 会产生噪声；需要在离线诊断里检查异常场景。

## 下一步

1. 实现 pose-angle subset builder。
2. 在 0007 的 1000-scene manifest 上跑离线覆盖诊断。
3. 复用 0007 swap labels 做弱代理评估。
4. 只有前两步有信号，再启动 50-scene VGGT smoke。
