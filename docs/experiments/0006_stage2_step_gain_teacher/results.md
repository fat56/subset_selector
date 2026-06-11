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
