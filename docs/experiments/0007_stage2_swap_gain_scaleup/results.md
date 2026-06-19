# 结果

## 摘要

`0007` 已完成从 `0006` direct swap-gain regression 到 `1000 scenes x 8 single swaps` 的扩规模实验。Teacher oracle 仍然很强，但 image-only student 没有变得更稳定。按验证集选择阈值后，held-out test mean delta 为 `-0.1703`，median 为 `-0.0203`，只有 `2/5` 个 seed 为正，因此本轮不晋级。

主要失败模式是阈值/泛化不稳定，而不是 teacher headroom 不足。Best single swap 在 `91.2%` 的场景中优于 `uniform20`，但 learned gain model 在 test 上的 sign accuracy 和 pairwise accuracy 都接近随机水平，并且有两个 seed 在选择较宽松验证阈值后发生明显 over-swap。

## 运行记录

| Run ID | 场景数 | 每场景 swap 数 | Feature | Seeds | 状态 |
|---|---:|---:|---|---|---|
| `swapgain1000_single8_smoke20` | 20 | 8 | DINOv2-S global | n/a | 完成 |
| `swapgain1000_single8` | 1000 | 8 | DINOv2-S global | n/a | 完成 |
| `swap_gain_regressor_global_dino_seed20260619-20260623` | 1000 | 8 | DINOv2-S global | 5 | 完成，未晋级 |

## 空间占用

`0007` cache run 前 `/` 上约 `814G` 可用。标签与训练完成后的最终状态：

| 产物 | 实际大小 |
|---|---:|
| Full VGGT cache: `swapgain1000_single8_images512` | `596G` |
| Smoke VGGT cache: `swapgain1000_single8_smoke20_images512` | `11G` |
| DINO image-only feature cache | `96M` |
| Run outputs、logs、labels、checkpoints | `236M` |
| `/` 最终剩余空间 | `208G` |

原先 `343G` 的估计偏低，因为 full run 除 single-swap subset 外，还包含 full/reference 相关工作。训练本身几乎没有带来空间压力；真正的大头是已完成的 VGGT tensor cache。

## Teacher 诊断

Full label run 从 `1000` 个场景生成了 `9000` 行标签，数据集配比为 `500` 个 DL3DV 和 `500` 个 WildRGBD。过程中修复了 VGGT cache runner 对 truncated image 的容忍后，任务完整跑完。

| 指标 | 数值 |
|---|---:|
| 至少有一个 swap candidate 的场景数 | `1000` |
| Label rows | `9000` |
| Swap oracle rate vs `uniform20` | `0.912` |
| Uniform minus best swap mean | `2.2821` |
| Uniform minus best swap min | `-2.1469` |
| Uniform minus best swap max | `8.5501` |
| Pair target gain mean | `-0.5667` |
| Pair target gain positive fraction | `0.4209` |

Oracle family 计数：

| Oracle family | 场景数 |
|---|---:|
| `swapgain20` | `912` |
| `uniform20` | `88` |

## Student 结果

主决策使用每个 seed 的 validation-selected threshold，并报告对应 held-out test 结果。

| Seed | Val threshold | Val delta | Test delta | Test win | Test deviation | Gain MAE | Sign acc | Pair acc | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260619 | `2.0` | `+0.0968` | `+0.0476` | `0.02` | `0.03` | `2.1512` | `0.5131` | `0.5289` | 保守，略正。 |
| 20260620 | `2.0` | `+0.0372` | `-0.0203` | `0.02` | `0.05` | `2.0353` | `0.5707` | `0.5364` | 接近中性，略负。 |
| 20260621 | `0.8` | `+0.1908` | `-0.5225` | `0.18` | `0.58` | `2.0925` | `0.5316` | `0.5127` | Test 上严重 over-swap。 |
| 20260622 | `1.5` | `+0.0736` | `+0.0379` | `0.18` | `0.40` | `2.1345` | `0.5386` | `0.5115` | 为正，但 deviation 偏高。 |
| 20260623 | `1.0` | `+0.2064` | `-0.3941` | `0.15` | `0.37` | `2.0072` | `0.5296` | `0.5217` | Test 上 over-swap。 |

汇总：

| 指标 | 数值 |
|---|---:|
| Mean test delta | `-0.1703` |
| Median test delta | `-0.0203` |
| Worst seed | `-0.5225` |
| Best seed | `+0.0476` |
| Positive seeds | `2/5` |
| Mean test win rate | `0.1100` |
| Mean test deviation rate | `0.2860` |
| Mean gain MAE | `2.0841` |
| Mean sign accuracy | `0.5367` |
| Mean pairwise accuracy | `0.5223` |

每个 seed 按 validation-selected threshold 得到的 dataset-wise test delta：

| 数据集 | Mean delta | Median delta | Positive seeds | Worst | Best |
|---|---:|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | `-0.2014` | `-0.0100` | `2/5` | `-0.8448` | `+0.1283` |
| `wildrgbd_harrison` | `-0.1391` | `-0.0306` | `2/5` | `-0.7336` | `+0.0645` |

Test-oracle threshold scan 仅用于诊断，不参与晋级判断。它在 seed `20260619` 上找到 `+0.2249`，在 seed `20260622` 上找到 `+0.0491`，其余三个 seed 的最佳 test-oracle 行为都是退回 `uniform20`，delta 为 `0.0`。

## 晋级总结

- Mean test delta: `-0.1703`，低于要求的 `+0.05`。
- Median test delta: `-0.0203`，低于要求的 `> 0`。
- Positive seeds: `2/5`，低于要求的 `4/5`。
- Worst seed: `-0.5225`，低于要求的 `>= -0.02`。
- Dataset-wise 结果在 DL3DV 和 WildRGBD 上都为负。
- 决策：不晋级；保留标签，但不要继续扩同类 direct global-DINO setup。

## 观察

- Teacher 标签是有价值的：swap oracle headroom 明显高于 student 实际收益。
- 扩大数据量没有解决 image-only student 的核心问题。模型能很好拟合 training ranking，但 validation-selected threshold 在 held-out scenes 上不可靠。
- `0006` 中的 split 敏感性不只是小数据噪声。到了 1000 个场景后，gain regressor 的 test ranking/sign calibration 仍然弱。
- 训练期间磁盘占用稳定，但 `596G` full VGGT cache 现在是主要空间风险。由于 labels 已经写出，这份 cache 对后续 student training 不是必需，只在需要复算 teacher 时有用。
