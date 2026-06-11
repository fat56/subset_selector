# 运行手册

## Smoke dense single-swap labels

第一版复用 `scripts/run_stage2_image_only_swap_gain_labels.py`，通过增加 `--single-swaps` 并关闭 multi-swap 来生成更密的 single-step labels。

```bash
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_swap_gain_labels.py \
  --run-dir runs/0006_stage2_step_gain_teacher/smoke_stepgain4 \
  --cache-root caches/vggt_omega/0006_stage2_step_gain_teacher/smoke_stepgain4_images512 \
  --limit-scenes 4 \
  --single-swaps 8 \
  --multi-swaps "" \
  --cache-devices cuda:0,cuda:1 \
  --max-pixels-per-image 1024 \
  --max-pointmap-points 60000
```

Smoke 结果：

- Scenes: `4`
- VGGT cache jobs: `32 / 32`
- Augmented labels: `92`
- `uniform_minus_best_swap_mean = +0.8152`
- `swap_best_win_rate_vs_uniform = 1.0000`

## Full300 dense single-swap labels

正式第一版仍使用 300 scenes，每个 scene `8` 个 single-swap candidates：

```bash
tmux new-session -d -s selector0006_stepgain300 '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_swap_gain_labels.py \
  --run-dir runs/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300 \
  --cache-root caches/vggt_omega/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300_images512 \
  --limit-scenes 300 \
  --single-swaps 8 \
  --multi-swaps "" \
  --cache-devices cuda:0,cuda:1 \
  --max-pixels-per-image 1024 \
  --max-pointmap-points 60000 \
  2>&1 | tee runs/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300/tmux.log
'
```

预估：

- VGGT cache jobs: `2400`
- 磁盘: 约 `100G`
- 若按 0005 sparse swap-gain 速度线性估计，双卡运行约数小时内完成。

实际结果：

- VGGT cache jobs: `2400 / 2400`
- Augmented labels: `6900` rows
- Cache size: `103G`
- `uniform_minus_best_swap_mean = +0.5982`
- `swap_best_win_rate_vs_uniform = 0.9067`
- `swap_oracle_rate = 0.3600`

## Gate-head student

第一轮 student 复用 0005 explicit gate-head，用 global DINO image-only feature 训练 candidate-level advantage/gate：

```bash
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_gate_head_training.py \
  --labels-csv runs/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300/augmented_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300/augmented_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0006_stage2_step_gain_teacher/gate_head_single8_global_dino_aw1_gw05_seed20260613 \
  --train-devices cuda:0 \
  --candidate-tag 20 \
  --seed 20260613 \
  --epochs 120 \
  --batch-size 32 \
  --lr 2e-4 \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --dropout 0.1 \
  --advantage-weight 1.0 \
  --gate-weight 0.5 \
  --rank-weight 0.0 \
  --positive-margin 0.2 \
  --log-every-steps 20
```

已完成 seeds：

- `20260609`
- `20260610`
- `20260611`
- `20260612`
- `20260613`

结论：

- Val-selected test Δ mean: `-0.0687`
- Val-selected 正 seed 数: `2 / 5`
- Test-oracle threshold Δ mean: `+0.1221`
- Test-oracle threshold 正 seed 数: `5 / 5`

因此 dense single-swap teacher 是强的，但 candidate-level gate-head 仍不稳。下一步改为 frame-level marginal gain regressor。

## Frame-score ridge-gain student

用 dense single-swap labels 复跑 `memory_frame_score + ridge_gain`。这一版仍只使用 global DINO image-only feature：

```bash
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300/augmented_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300/augmented_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0006_stage2_step_gain_teacher/frame_score_single8_ridge_w02_seed20260609 \
  --model-kind memory_frame_score \
  --candidate-tag 20 \
  --seed 20260609 \
  --epochs 120 \
  --batch-size 32 \
  --lr 2e-4 \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --dropout 0.1 \
  --rank-weight 1.0 \
  --ce-weight 0.3 \
  --coverage-weight 0.05 \
  --uniform-gate-margin 0.2 \
  --uniform-advantage-weight 0.5 \
  --uniform-advantage-scale 1.0 \
  --uniform-advantage-clip 4.0 \
  --frame-target-mode ridge_gain \
  --frame-target-weight 0.2 \
  --frame-target-ridge 0.01 \
  --frame-target-clip 5.0 \
  --train-devices cuda:0 \
  --log-every-steps 20
```

`frame-target-weight=0.5` 的对照只改 `--run-dir`、`--frame-target-weight 0.5` 和 `--train-devices cuda:1`。

结果：

| Run | Val Δ | Test Δ | Post-hoc test-oracle Δ | 判断 |
|---|---:|---:|---:|---|
| `frame_score_single8_ridge_w02_seed20260609` | `+0.0367` | `-0.0420` | `+0.0815` | val 小正，test 负 |
| `frame_score_single8_ridge_w05_seed20260609` | `+0.0610` | `-0.0793` | `+0.0239` | val 小正，test 负 |

结论：dense labels 没有让 ridge-gain frame-score 变成稳定 selector。该分支先停。

## Patch-summary temporal gate-head

把输入换成 0005 Main V6 的 patch-summary temporal DINO feature：

```bash
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_gate_head_training.py \
  --labels-csv runs/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300/augmented_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0006_stage2_step_gain_teacher/stepgain_uniform20_dinov2_single8_300/augmented_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_patch_summary_temporal \
  --run-dir runs/0006_stage2_step_gain_teacher/gate_head_single8_patch_temporal_aw1_gw05_seed20260609 \
  --train-devices cuda:0 \
  --candidate-tag 20 \
  --seed 20260609 \
  --epochs 120 \
  --batch-size 32 \
  --lr 2e-4 \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --dropout 0.1 \
  --advantage-weight 1.0 \
  --gate-weight 0.5 \
  --rank-weight 0.0 \
  --positive-margin 0.2 \
  --log-every-steps 20
```

已完成 seeds：

| Seed | Val Δ | Test Δ | Test-oracle Δ | 判断 |
|---:|---:|---:|---:|---|
| `20260609` | `+0.0496` | `-0.0923` | `0.0000` | test 负 |
| `20260610` | `+0.0462` | `+0.0420` | `+0.1294` | 小正 |

结论：patch-summary temporal feature 有一点正信号，但 `1/2` seed 为正，不足以作为结果。下一步使用同一输入尝试 `rank_weight > 0`。
