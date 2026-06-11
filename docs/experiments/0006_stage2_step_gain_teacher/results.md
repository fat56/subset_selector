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
- 下一步不再继续加 seed 堆 gate-head，而是训练真正的 frame-level marginal gain student：输入每帧 image-only feature，直接回归该帧进入 `uniform20` base subset 的单步收益，并用 predicted gain 形成 replacement/topK。
