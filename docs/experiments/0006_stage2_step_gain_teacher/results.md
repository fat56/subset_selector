# 结果

## Smoke: `smoke_stepgain4`

- 运行目录: `runs/0006_stage2_step_gain_teacher/smoke_stepgain4`
- Cache root: `caches/vggt_omega/0006_stage2_step_gain_teacher/smoke_stepgain4_images512`
- Scenes: `4`
- Candidate: 每个 scene 生成 `8` 个 `swapgain20_dino1_rank000-007` single-swap candidates
- VGGT cache jobs: `32 / 32` 成功
- Augmented labels: `92` rows
- Cache size: `1.4G`

诊断：

| 指标 | 数值 |
|---|---:|
| `swap_best_win_rate_vs_uniform` | `1.0000` |
| `swap_oracle_rate` | `0.2500` |
| `uniform_minus_best_swap_mean` | `+0.8152` |
| `uniform_minus_best_swap_min` | `+0.4043` |
| `uniform_minus_best_swap_max` | `+1.4598` |

Oracle family:

| Family | Scenes |
|---|---:|
| `uniform_jitter20` | `3` |
| `swapgain20` | `1` |

解读：

- 4/4 个 smoke scenes 都能在 8 个 single-swap candidates 中找到优于 `uniform20` 的替换。
- 这说明 dense single-swap teacher 比 0005 的 sparse 4-candidate swap-gain 更适合继续放大。
- 但 smoke 的 oracle 仍多数是 `uniform_jitter20`，因此 full300 后仍需看 student 是否能学会稳定 gate，而不是只看 teacher headroom。

## Full300 dense single-swap labels

- 运行目录: `runs/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300`
- Cache root: `caches/vggt_omega/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300_images512`
- Scenes: `300`
- 数据分布: `DL3DV-ALL-480P=150` scenes，`wildrgbd_harrison=150` scenes
- Candidate: 每个 scene 生成 `8` 个 `swapgain20_dino1_rank000-007` single-swap candidates
- VGGT cache jobs: `2400 / 2400` 成功
- Augmented labels: `6900` rows
- Cache size: `103G`

Teacher 诊断：

| 指标 | 数值 |
|---|---:|
| `swap_best_win_rate_vs_uniform` | `0.9067` |
| `swap_oracle_rate` | `0.3600` |
| `uniform_minus_best_swap_mean` | `+0.5982` |
| `uniform_minus_best_swap_min` | `-1.1073` |
| `uniform_minus_best_swap_max` | `+3.3518` |

Full300 的 dense single-swap teacher 明显比 0005 sparse swap-gain 更强：90.7% scenes 可以在 8 个单步替换中找到优于 `uniform20` 的 subset，且 best swap 平均带来 `+0.5982` 的 `target_error` 改善。不过 oracle 仍主要落在 `uniform_jitter20` 和 `swapgain20` 两类上，说明 teacher headroom 不等同于 student 已经能稳定选中这些替换。

Oracle family：

| Family | Scenes |
|---|---:|
| `uniform_jitter20` | `143` |
| `swapgain20` | `108` |
| `random20` | `12` |
| `convnext_kcenter20` | `12` |
| `uniform20` | `11` |
| `dinov2_kcenter20` | `9` |
| `motion_spread20` | `5` |

## Gate-head student: global DINO, 5 seeds

第一轮 student 仍复用 0005 的 explicit gate-head：输入 image-only global DINO feature，输出候选 subset 的 advantage/gate 分数，再通过 validation 选择 mode 和 threshold。这里 student 推理不读取 VGGT token/output。

| Seed | Val 选择规则 | Val Δ | Test Δ | Test win | Test deviation | Test 选择分布 | Test-oracle Δ |
|---:|---|---:|---:|---:|---:|---|---:|
| `20260609` | `gate_logit@0.0` | `+0.0576` | `-0.1469` | `0.200` | `0.433` | `motion_spread20=1, swapgain20=5, uniform20=17, uniform_jitter20=7` | `+0.0004` |
| `20260610` | `advantage@0.05` | `+0.1030` | `+0.1250` | `0.333` | `0.600` | `swapgain20=3, uniform20=12, uniform_jitter20=15` | `+0.2855` |
| `20260611` | `gate_logit@-0.5` | `+0.1308` | `+0.0028` | `0.200` | `0.267` | `motion_spread20=1, swapgain20=4, uniform20=22, uniform_jitter20=3` | `+0.2173` |
| `20260612` | `advantage@-0.2` | `+0.1529` | `-0.0791` | `0.433` | `0.967` | `swapgain20=29, uniform20=1` | `+0.0910` |
| `20260613` | `advantage@0.1` | `+0.1835` | `-0.2455` | `0.100` | `0.367` | `motion_spread20=3, swapgain20=6, uniform20=19, uniform_jitter20=2` | `+0.0161` |

汇总：

| 指标 | 数值 |
|---|---:|
| Val-selected test Δ mean | `-0.0687` |
| Val-selected test Δ median | `-0.0791` |
| Val-selected 正 seed 数 | `2 / 5` |
| Val-selected test Δ min/max | `-0.2455 / +0.1250` |
| Test-oracle threshold Δ mean | `+0.1221` |
| Test-oracle threshold 正 seed 数 | `5 / 5` |

解读：

- Dense single-swap label 本身是有信号的，teacher 端比 `uniform20` 的可提升空间很明显。
- 现有 gate-head 的 validation 分数全为正，但 val-selected rule 在 test 上只有 `2/5` 为正，说明主要问题是 calibration/threshold selection 和 split sensitivity。
- Test-oracle threshold 在 `5/5` seeds 上为正，表示模型分数里确实有可用排序信号；只是当前候选级 gate 不能可靠地把这个信号转成固定的推理规则。

## Frame-score ridge-gain student

第二轮改用 `memory_frame_score`：模型输出每帧 score，候选 subset 的 score 是 selected frames 平均值；同时用 `ridge_gain` 把 candidate-level utility 近似分摊到 frames。输入仍为 image-only global DINO feature。

| Run | Frame target weight | Val Δ | Test Δ | Post-hoc test-oracle Δ | Test 选择分布 |
|---|---:|---:|---:|---:|---|
| `frame_score_single8_ridge_w02_seed20260609` | `0.20` | `+0.0367` | `-0.0420` | `+0.0815` | `uniform20=23, swapgain20=7` |
| `frame_score_single8_ridge_w05_seed20260609` | `0.50` | `+0.0610` | `-0.0793` | `+0.0239` | `uniform20=23, swapgain20=7` |

解读：

- Dense single-swap labels 确实让 frame-score 在 validation 上出现小正信号。
- 但 val-selected checkpoint/rule 在 test 上仍为负，post-hoc scan 只有 test-oracle 能扫出小正。
- 这说明 `ridge_gain` 这种把 subset utility 线性分摊到 frames 的近似仍然不足；它能学到一点排序，但不能形成稳定 selector。

## Gate-head student: patch-summary temporal DINO

第三轮把输入从 global DINO feature 换成 0005 Main V6 的 patch-summary temporal feature：`frame_features=2304`，`image_stats=16`。模型仍是 explicit gate-head，标签使用 0006 dense single-swap labels。

| Seed | Val 选择规则 | Val Δ | Test Δ | Test win | Test deviation | Test 选择分布 | Test-oracle Δ |
|---:|---|---:|---:|---:|---:|---|---:|
| `20260609` | `gate_logit@-0.5` | `+0.0496` | `-0.0923` | `0.033` | `0.167` | `uniform20=25, swapgain20=5` | `0.0000` |
| `20260610` | `gate_logit@-0.5` | `+0.0462` | `+0.0420` | `0.100` | `0.233` | `uniform20=23, swapgain20=7` | `+0.1294` |

汇总：

| 指标 | 数值 |
|---|---:|
| Val-selected test Δ mean | `-0.0251` |
| Val-selected 正 seed 数 | `1 / 2` |
| Test-oracle threshold Δ mean | `+0.0647` |

解读：

- Patch-summary temporal feature 比 frame-score ridge-gain 略有希望，至少 seed `20260610` 在 val-selected test 上为正。
- 但 seed `20260609` 仍为负，且 test-oracle 均值低于 global DINO gate-head 的 `+0.1221`。
- 目前不能把 patch-temporal gate 视为稳定成果。下一步改为在 gate-head 中加入 pairwise ranking loss，强制模型保留 dense candidates 间的顺序信息，而不是只做 advantage/gate 回归。

## Gate-head student: patch-summary temporal + rank loss

第四轮保持 patch-summary temporal DINO 输入，在 explicit gate-head 中加入 `rank_weight=0.2`，希望保留 dense candidates 之间的 pairwise order。

| Seed | Val 选择规则 | Val Δ | Test Δ | Test win | Test deviation | Test 选择分布 | Test-oracle Δ |
|---:|---|---:|---:|---:|---:|---|---:|
| `20260609` | `advantage@0.05` | `+0.0571` | `-0.2408` | `0.200` | `0.667` | `uniform20=10, swapgain20=20` | `0.0000` |
| `20260610` | `advantage@1.0` | `+0.0263` | `+0.0173` | `0.033` | `0.033` | `uniform20=29, uniform_jitter20=1` | `+0.2135` |

解读：

- `rank_weight=0.2` 没有稳定改善 val-selected test，seed `20260609` 被 val 选到过激规则，test 明显变差。
- 但 seed `20260610` 的 test-oracle 达到 `+0.2135`，说明 rank loss 可能增强了部分排序信号，只是校准更不稳。

## Conservative validation selection diagnostic

不重训，直接在已有 `gate_scan.json` 上试更保守的 validation 选择规则：

```text
score = val_uniform_minus_learned_error - lambda * val_deviation_rate
```

诊断结果：

| Run family | 规则 | Test Δ mean | 正 seed 数 | 备注 |
|---|---|---:|---:|---|
| global DINO gate-head | 原始 val max | `-0.0687` | `2 / 5` | 易选过激规则 |
| global DINO gate-head | `lambda=0.10` | `-0.0073` | `4 / 5` | 接近零，但均值仍负 |
| patch temporal gate-head | `lambda=0.10` | `-0.0251` | `1 / 2` | 无改善 |
| patch temporal + rank02 | `lambda=0.10` | `+0.0087` | `1 / 2` | 主要靠回到 uniform，提升太弱 |

结论：

- 问题确实很大一部分是 validation rule 过激；对 global DINO，加入 deviation penalty 能明显改善正 seed 数。
- 但保守选择规则目前只能把结果拉到接近零，不能算达成稳定 selector。
- 因此下一步验证 global DINO gate-head 加 `rank_weight=0.2`，再尝试更直接的 frame-level swap preference。

## Gate-head student: global DINO + rank loss

第五轮回到 test-oracle signal 最强的 global DINO gate-head，在 explicit gate-head 中加入 `rank_weight=0.2`。

| Seed | Val 选择规则 | Val Δ | Test Δ | Test win | Test deviation | Test 选择分布 | Test-oracle Δ |
|---:|---|---:|---:|---:|---:|---|---:|
| `20260609` | `gate_logit@-0.5` | `+0.0641` | `-0.1311` | `0.333` | `0.600` | `uniform_jitter20=11, uniform20=12, swapgain20=6, motion_spread20=1` | `+0.0306` |
| `20260610` | `advantage@0.0` | `+0.1026` | `-0.0183` | `0.267` | `0.700` | `uniform20=9, uniform_jitter20=7, swapgain20=14` | `+0.0514` |

汇总：

| 指标 | 数值 |
|---|---:|
| Val-selected test Δ mean | `-0.0747` |
| Val-selected 正 seed 数 | `0 / 2` |
| Test-oracle threshold Δ mean | `+0.0410` |

解读：

- global DINO 加 rank loss 没有继承原始 global DINO gate-head 的 test-oracle 强信号，val-selected 两个 seed 都为负。
- rank loss 让 validation 选择更激进，test 端仍然容易选到 bad swap/jitter。
- 该分支先停，不继续扩大 seed。

## Frame-score student: swap-pair preference

第六轮在 `memory_frame_score` 上新增 `swap_pair` frame preference：对每个 `swapgain20_dino1_rank*` single-swap，恢复 added frame 和 removed uniform frame；如果该 swap 相对 `uniform20` 改善 `target_error`，监督 `score(added) > score(removed)`，反之则监督相反方向。输入仍为 image-only global DINO feature。

| Seed | Val Δ | Test Δ | Test win | Test 选择分布 | Uniform-margin val-selected Test Δ | Post-hoc margin Test-oracle Δ |
|---:|---:|---:|---:|---|---:|---:|
| `20260609` | `+0.0431` | `-0.1346` | `0.067` | `uniform20=21, swapgain20=9` | `-0.5690` | `+0.0259` |
| `20260610` | `+0.0286` | `+0.0388` | `0.133` | `uniform20=20, swapgain20=10` | `+0.0388` | `+0.0502` |

解读：

- `swap_pair` 能把选择分布收敛到 `uniform20/swapgain20`，比 earlier frame-score 更贴近 single-swap teacher。
- 但 test 仍是 `1/2` seed 为正，均值约 `-0.0479`，不能算稳定成果。
- Uniform-margin scan 没有稳定救回坏 seed；说明问题不只是阈值，而是 bad swap 的排序本身还不稳。
- 下一步应该把训练/eval 候选先收窄到 `uniform20 + swapgain20`，避免 random/jitter/contiguous 候选的 candidate-level loss 干扰 single-swap frame preference。
