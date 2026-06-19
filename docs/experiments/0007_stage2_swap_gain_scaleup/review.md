# 复盘

## 决策

不晋级。`0007` 应保留为有价值的 teacher-label 资产，但 direct global-DINO swap-gain regressor 不应作为下一版 selector policy 晋级。

这次扩规模清楚回答了核心问题：单纯增加场景数量，没有让 `0006` 的信号变稳定。Teacher 仍然有很强的 oracle headroom，但 student 不能可靠判断 held-out scenes 上什么时候应该 swap。

## 问题回答

| 问题 | 回答 |
|---|---|
| 从 300 扩到 1000 个场景后，validation-to-test calibration 是否变好？ | 没有。多个 seed 的 validation-selected threshold 发生过拟合；mean test delta 为 `-0.1703`。 |
| Direct gain regression 是否仍是目前最强的 student 形式？ | 它仍是目前测过的最好路线，但这轮说明只用 global DINO image features 时还不够稳健。 |
| WildRGBD 与 DL3DV 上收益是否均衡？ | 没有任何一边有稳定收益。DL3DV mean test delta 为 `-0.2014`；WildRGBD mean test delta 为 `-0.1391`。 |
| 扩规模后 teacher oracle 是否仍然强？ | 是。Best swap 在 `91.2%` 的场景中优于 `uniform20`，平均提升 `+2.2821`。 |
| 磁盘使用是否符合计划？ | 训练部分没问题，但 VGGT cache 超出估计：full cache `596G`，最终剩余空间 `208G`。 |

## 证据

- 实验方案: `docs/experiments/0007_stage2_swap_gain_scaleup/proposal.md`
- 运行手册: `docs/experiments/0007_stage2_swap_gain_scaleup/runbook.md`
- 实验结果: `docs/experiments/0007_stage2_swap_gain_scaleup/results.md`
- 实验配置: `configs/experiments/0007_stage2_swap_gain_scaleup.yaml`
- 完整标签: `runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8/augmented_hardlabel_train_labels.csv`
- Student 汇总: `runs/0007_stage2_swap_gain_scaleup/swap_gain_regressor_global_dino_seed*/summary.json`

## 解读

Teacher 侧不是瓶颈。8 个 single-swap candidate 经常显著优于 `uniform20`，但很多单个 swap 也是有害的，而 global image-only student 没有学到可靠的场景级 accept/reject 规则。它可以拟合 training ranking，但 held-out sign accuracy 和 pairwise accuracy 仍接近随机水平。

两个最差 seed 选择了较低验证阈值（`0.8` 和 `1.0`），把 test deviation 提高到 `0.58` 和 `0.37`，导致严重损失。更保守的阈值避免了灾难性负收益，但多数情况下只是退回 `uniform20`，实际正收益很小。

## 下一步

- 在改变 student 信号之前，不再启动同形态的 1000 场景 global-DINO direct-gain run。
- 下一步尝试更强的 image-only feature：patch-summary/temporal aggregation、相邻帧 motion cues，或带显式 uniform-subset context 的轻量 per-frame sequence model。
- 增加保守 calibration objective 或 threshold policy，让 false-positive swap 的代价高于 missed swap。
- 复用 `0007` 标签做 ablation；除非测试新的 candidate family，否则避免重新生成 VGGT 标签。
- 如果磁盘压力再次出现，在保留 labels、jobs、summaries 与 DINO feature cache 后，可以删除 `596G` full VGGT cache。
