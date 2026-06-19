# 1000 场景 Swap-Gain 扩规模实验

## 元信息

- 实验 ID: `0007_stage2_swap_gain_scaleup`
- 阶段: `stage2`
- 状态: 已完成，未晋级
- 创建时间: 2026-06-19
- 配置: `configs/experiments/0007_stage2_swap_gain_scaleup.yaml`

## 问题

当 teacher 标签集从 `300 scenes x 8 single swaps` 扩大到约 `1000 scenes x 8 single swaps` 后，`0006` 中直接 swap-gain 回归呈现出的弱但可复现信号，是否会变得明显更稳定？

## 假设

`0006` 第一次得到一个按验证集选择阈值后，在 held-out test 上相对 `uniform20` 有正平均收益的 image-only student：test delta 均值 `+0.0535`，`4/5` 个 seed 为正，但中位数只有 `+0.0082`。当时看起来瓶颈并不是 teacher headroom；密集 single-swap 标签在 `90.7%` 的场景中优于 `uniform20`。更可能的瓶颈是场景数太少导致 student 泛化与阈值校准不稳。

因此，本实验假设：扩展到约 1000 个场景后，split 敏感性会下降，直接 gain regressor 的验证集阈值选择会更可靠。如果这个假设成立，不依赖 test-oracle threshold scan 的情况下，多 seed test delta 也应该稳定为正。

最终结果：该假设没有得到支持。Teacher oracle 依然很强，但按验证集选择的 student 阈值在 test 上得到 mean delta `-0.1703`、median `-0.0203`，只有 `2/5` 个 seed 为正。

## 方法

实验沿用 `0006` 中最成功的形式：

```text
base subset = uniform20(scene)
candidate = uniform20 - removed_frame + added_frame
gain = target_error(uniform20) - target_error(candidate)
student predicts gain from image-only features
```

第一轮扩规模目标：

- `1000` 个场景。
- 每个场景 `8` 个 single-swap candidate。
- `20%` 固定 K 预算。
- VGGT-OMEGA 只用于离线 teacher 标签。
- Student 输入仍然是 image-only DINO feature 加图像统计量。

重要实现说明：`0006` 的标签生成流程绑定了 `hardlabel300` / `richer300` 输入。真正的 1000 场景实验需要先构建 1000 场景 source manifest 和 image-only feature cache，不能假设只给旧脚本加 `--limit-scenes 1000` 就足够。

## 数据计划

计划的 1000 场景来源配比：

| 数据集 | 场景数 | 原因 |
|---|---:|---|
| WildRGBD Harrison | 500 | 以物体为中心，带传感器深度；与 `0003/0006` 保持连续性。 |
| DL3DV | 500 | 更丰富的真实视频场景；降低对 WildRGBD 物体扫描数据的过拟合。 |

LTM30 validation scenes 必须继续排除。场景级 split 为 `80/10/10`，按数据集分层。

如果某个数据源过滤后数量不足，则从另一个数据源补足到接近 1000，并在 `results.md` 中记录最终配比。

## 标签生成

对每个入选场景：

1. 使用与 hard-label 实验相同的策略采样 full frame set：
   - WildRGBD: 最多 `100` 帧。
   - DL3DV: 最多 `80` 帧。
2. 构建 `full` 和 `uniform20` image list。
3. 使用 DINOv2 image-only feature 对 `uniform20` 之外的候选 add frame 排序。
4. 生成 `8` 个 single-swap candidate: `swapgain20_dino1_rank000-007`。
5. 按需运行 full/uniform/swap candidate 的 VGGT-OMEGA cache。
6. 计算 VGGT-native depth/pose/point-map 指标。
7. 写出 `augmented_hardlabel_train_labels.csv` 和 `augmented_cache_jobs.json`。

如果某个场景已经有 full/uniform metrics，则复用；否则纳入本实验 cache 预算。

## Student 训练

主模型：

- 脚本: `scripts/run_stage2_image_only_swap_gain_regressor.py`
- Feature cache: DINOv2-S/ViT-S global image-only features。
- Seeds: `20260619`, `20260620`, `20260621`, `20260622`, `20260623`。
- Epochs: `120`。
- Objective: gain regression + sign loss + pairwise rank loss。
- 阈值选择: 只使用 validation-selected threshold；test-oracle scan 只作为诊断。

只有主模型通过或非常接近通过时，才考虑二级模型：

- Patch-summary temporal DINO features。
- 相同的 direct gain objective。

## 指标

主指标：

- 5 个 seed 的 `test_uniform_minus_learned_error_mean`。

晋级标准：

- Mean test delta `>= +0.05`。
- Median test delta `> 0`。
- 至少 `4/5` 个 seed 为正。
- Worst seed `>= -0.02`。
- Pass/fail 只看 val-selected rule，不使用 test-oracle threshold。

辅助指标：

- Test win rate vs `uniform20`。
- Test deviation rate。
- Gain MAE。
- Gain sign accuracy。
- Pairwise gain accuracy。
- Test-oracle threshold delta，仅用于诊断。
- WildRGBD 与 DL3DV 的 dataset-wise delta。

## 资源预算

计划前磁盘状态：`/` 上约 `814G` 可用。

`0006` 观察到的 cache 成本：

- `300 scenes x 8 single swaps` 使用约 `103G`。
- 线性估计 `1000 scenes x 8 single swaps` 约 `343G`。

计划预算：

| 组件 | 估计 |
|---|---:|
| 新 single-swap VGGT cache | `340G` |
| 如果不能复用 full/uniform reference cache | 额外占用，由 smoke 估计 |
| Image-only DINO features | `< 2G` |
| Run outputs/checkpoints/logs | `< 5G` |
| Cache 后安全余量 | 目标 `>= 250G` 可用 |

如果第一个 smoke 证明单场景 cache 成本接近 `0006` 估计，则该实验空间上可行。如果可用空间低于 `250G`，停止继续启动 VGGT cache batch，并清理已完成 cache 或降低规模。

## 决策规则

通过：

- Direct gain regressor 满足上述主晋级标准。
- Dataset-wise 结果没有出现一个数据源贡献全部收益、另一个数据源明显为负的情况。

继续研究但不晋级：

- Mean test delta 为正，但 median 接近 0 或 seed 敏感性仍然高。
- 这种情况下保留标签，尝试更强的 image-only feature 或更保守的 calibrated threshold objective。

失败：

- 最好的 validation-selected 策略退回到 `uniform20`，或者 mean test delta 非正。
- 如果 teacher oracle 依然很强，则失败原因在 student 侧；下一步应改变 feature/model，而不是继续生成更多同类标签。

## 风险

- 现有 label script 偏 hardlabel300；在 VGGT run 前可能需要 manifest-based 1000 场景 label builder。
- Full/uniform reference cache 可能比 `340G` single-swap 估计额外占更多空间。
- 更多数据可能降低方差，但不一定解决 feature 不足。
- Validation split 只有 100 个场景，阈值仍可能过拟合；需要保留 per-dataset validation summary。
