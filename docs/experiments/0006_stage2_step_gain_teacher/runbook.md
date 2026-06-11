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
