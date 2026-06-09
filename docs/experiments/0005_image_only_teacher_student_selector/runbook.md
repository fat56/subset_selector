# 运行手册

## 当前状态

`0005/Main V1` 已实现并完成 hardlabel300 第一轮：

- `main_v1_convnext_tiny_candidate_rank`
- `main_v1_dinov2_vits14_candidate_rank`

这轮没有得到超过 `uniform20` 的正提升。best-val checkpoint 都退回 `uniform20`，因此不推进 hard subset VGGT rerun。

## 核心约束

Student selector 推理时只允许使用：

- 原始 RGB 图像。
- cheap image backbone features，例如 DINOv2-S、MobileNetV3、ConvNeXt-Tiny。
- frame order / `frame_pos`。
- 轻量图像质量统计。

Student selector 推理时禁止使用：

- VGGT-OMEGA `camera_token`。
- VGGT-OMEGA `register_tokens`。
- full scene VGGT output。
- 任何需要先跑 full VGGT-OMEGA 才能得到的特征。

## Main V1 计划

目标：先训练 batch cheap-feature candidate selector。

步骤：

1. 为 hardlabel300 scenes 缓存 cheap image features。
2. 读取 `0003` hard-native candidate labels。
3. 用 candidate masks 聚合 cheap features，训练 candidate rank scorer。
4. 与 `uniform20`、random、cheap-feature k-center 对比。
5. 若 val/test 都无法超过 uniform，停止，不扩到 streaming。

## 已实现脚本

当前已实现：

```bash
scripts/prepare_stage2_image_only_selector_features.py
scripts/run_stage2_image_only_selector_training.py
scripts/run_stage2_image_only_richer_candidate_labels.py
```

暂未单独拆 `evaluate_stage2_image_only_selector.py`；训练脚本会在 best-val checkpoint 上输出 train/val/test summary。

## Smoke

ConvNeXt-Tiny smoke:

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/prepare_stage2_image_only_selector_features.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --out-dir caches/image_features/0005/smoke_convnext_tiny \
  --backbone convnext_tiny \
  --device cuda:0 \
  --batch-size 32 \
  --limit-scenes 12 \
  --force
```

训练 smoke:

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --feature-cache caches/image_features/0005/smoke_convnext_tiny \
  --run-dir runs/0005_image_only_teacher_student_selector/smoke_convnext_tiny \
  --model-kind memory_candidate_set \
  --hidden-dim 128 \
  --num-layers 1 \
  --num-heads 4 \
  --memory-slots 4 \
  --epochs 2 \
  --batch-size 4 \
  --limit-scenes 12 \
  --train-devices cuda:0 \
  --log-every-steps 1
```

## Main V1 正式运行

### ConvNeXt-Tiny

```bash
tmux new-session -d -s selector0005_convnext '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/prepare_stage2_image_only_selector_features.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --out-dir caches/image_features/0005/hardlabel300_convnext_tiny \
  --backbone convnext_tiny \
  --device cuda:0 \
  --batch-size 64 \
  2>&1 | tee runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_candidate_rank/feature_cache.log &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_convnext_tiny \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_candidate_rank \
  --model-kind memory_candidate_set \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 60 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --log-every-steps 10 \
  2>&1 | tee runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_candidate_rank/tmux.log
'
```

### DINOv2-S/ViT-S

```bash
tmux new-session -d -s selector0005_dinov2 '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/prepare_stage2_image_only_selector_features.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --out-dir caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --backbone dinov2_vits14 \
  --device cuda:1 \
  --batch-size 64 \
  2>&1 | tee runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_candidate_rank/feature_cache.log &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_candidate_rank \
  --model-kind memory_candidate_set \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 60 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --log-every-steps 10 \
  2>&1 | tee runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_candidate_rank/tmux.log
'
```

## Main V1 判定

通过条件：

- val `uniform_minus_learned_error > 0`。
- held-out test `uniform_minus_learned_error > 0`。
- student inference 不读取 VGGT features。

不通过条件：

- val 有提升但 test 为负，视为 calibration 失败。
- test oracle scan 最优为 uniform fallback，说明当前 student 不可靠。

## Richer Candidate Labels

### 4-scene smoke

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_richer_candidate_labels.py \
  --run-dir runs/0005_image_only_teacher_student_selector/richer_candidates_smoke4 \
  --cache-root caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_smoke4_images512 \
  --limit-scenes 4 \
  --cache-devices cuda:0,cuda:1 \
  --max-pixels-per-image 512 \
  --max-pointmap-points 30000
```

### 正式 hardlabel300 cache

正式 cache 状态：

- Total jobs: `2700`
- Ready jobs: `2700`
- Missing jobs: `0`
- Complete scenes: `300`
- Cache root: `caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_images512`
- Cache size: 约 `167G`

历史说明：cache 过程中曾因 GPU/driver 状态异常中断，恢复后通过 `selector0005_richer_resume` 补齐，两个 worker 均 `failed=0`。

如需重新 resume，命令仍可复用；脚本会跳过已经 `cache_ready` 的 jobs：

```bash
mkdir -p runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_resume
tmux new-session -d -s selector0005_richer_resume '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_richer_candidate_labels.py \
  --run-dir runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_resume \
  --cache-root caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_images512 \
  --cache-devices cuda:0,cuda:1 \
  --cache-only \
  2>&1 | tee runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_resume/tmux.log
'
```

缓存补全后，生成正式 merged labels：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_richer_candidate_labels.py \
  --run-dir runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300 \
  --cache-root caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_images512 \
  --skip-cache \
  --labels-only
```

当前正式输出：

- Labels: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_hardlabel_train_labels.csv`
- Jobs: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_cache_jobs.json`
- Diagnostic: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/full300_richer_diagnostic_summary.json`
- Rows: `4500`

### Ready-only partial 诊断

GPU 异常时，可以只分析已经完整缓存的 scene：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_richer_candidate_labels.py \
  --run-dir runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_ready108 \
  --cache-root caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_images512 \
  --skip-cache \
  --labels-only \
  --ready-only \
  --max-pixels-per-image 1024 \
  --max-pointmap-points 60000
```

当前 ready108 输出：

- Labels: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_ready108/merged_hardlabel_train_labels.csv`
- Jobs: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_ready108/merged_cache_jobs.json`
- Diagnostic: `partial_richer108_diagnostic_summary.json`

## Richer108 CPU 诊断

CUDA 不可用时的轻量诊断命令：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_ready108/merged_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_ready108/merged_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_richer108_cpu \
  --model-kind memory_candidate_set \
  --hidden-dim 128 \
  --num-layers 1 \
  --num-heads 4 \
  --memory-slots 4 \
  --epochs 40 \
  --batch-size 16 \
  --lr 2e-4 \
  --train-devices cpu \
  --log-every-steps 10
```

当前 DINOv2-S ready108 结果：

- Val `uniform_minus_learned_error = +0.2447`
- Test `uniform_minus_learned_error = +0.0436`
- 这只是 partial 正信号，不作为 promotion 结论。

## Richer300 正式训练

已完成 DINOv2-S：

```bash
tmux new-session -d -s selector0005_dinov2_richer300 '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_richer300 \
  --model-kind memory_candidate_set \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 60 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --log-every-steps 10 \
  2>&1 | tee runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_richer300/tmux.log
'
```

结果：

- Val `uniform_minus_learned_error = 0.0000`
- Test `uniform_minus_learned_error = 0.0000`
- Best checkpoint 选择 `uniform20` fallback。

已完成 ConvNeXt-Tiny 对照：

```bash
tmux new-session -d -s selector0005_convnext_richer300 '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_convnext_tiny \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_richer300 \
  --model-kind memory_candidate_set \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 60 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --log-every-steps 10 \
  2>&1 | tee runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_richer300/tmux.log
'
```

结果：

- Val `uniform_minus_learned_error = -0.0215`
- Test `uniform_minus_learned_error = 0.0000`
- Test 选择 `uniform20` fallback。

## Main V1 下一步: Uniform Fallback Gate

richer300 证明 label pool 已经变宽，但 `memory_candidate_set` top-1 deviation 不可靠。下一步优先做 gate，而不是继续换 backbone。

目标：

- 默认选 `uniform20`。
- 只有当最佳 non-uniform candidate 分数超过 `uniform20` 分数至少 `margin` 时才偏离。
- 在 val 上 sweep `margin`，用 test 验证是否真正超过 uniform。

建议 sweep:

```text
margin in [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]
```

通过条件：

- Val `uniform_minus_learned_error > 0`
- Test `uniform_minus_learned_error > 0`
- Test deviation rate 不为极低偶然值。

不通过则进入 Main V3 marginal-gain teacher。

当前 post-hoc gate sweep 已完成：

```bash
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/evaluate_stage2_image_only_selector_gate.py \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_richer300 \
  --device cuda:0 \
  --batch-size 32 \
  --out runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_richer300/uniform_gate_scan.json

PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/evaluate_stage2_image_only_selector_gate.py \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_richer300 \
  --device cuda:1 \
  --batch-size 32 \
  --out runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_richer300/uniform_gate_scan.json
```

结果：

- DINOv2-S val-selected gate: val `+0.0016`, test `-0.1196`
- ConvNeXt-Tiny val-selected gate: val `+0.0165`, test `0.0000`

判断：post-hoc gate 不足以稳定超过 uniform。下一步应改训练 loss。

## Main V1: Uniform-Gated Loss

已新增训练设置：

```text
uniform_gate_margin = 0.10 or 0.20
oracle CE only if oracle_error + margin < uniform_error
otherwise CE target = uniform20
pairwise loss keeps all clear pairwise preferences
```

实际运行时用了本地队列脚本作为 run artifact：

```bash
bash runs/0005_image_only_teacher_student_selector/run_gated_queue.sh
```

该脚本不进入 git；核心命令如下。优先跑 DINOv2-S，因为 ready108 和 richer300 pairwise accuracy 都略好：

```bash
tmux new-session -d -s selector0005_gated_queue '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_richer300_gated \
  --model-kind memory_candidate_set \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 60 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --uniform-gate-margin 0.2 \
  --log-every-steps 10 \
  2>&1 | tee runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_richer300_gated/tmux.log
'
```

完成 runs：

| Run | Margin | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---:|---:|---:|---|
| `main_v1_dinov2_vits14_richer300_gated_m02` | `0.2` | `+0.0005` | `+0.0328` | 当前最好，但幅度小 |
| `main_v1_dinov2_vits14_richer300_gated_m05` | `0.5` | `+0.0166` | `-0.6146` | 失败 |
| `main_v1_convnext_tiny_richer300_gated_m02` | `0.2` | `0.0000` | `0.0000` | 退回 uniform |

结论：Main V1 有小正信号，但不够稳定。下一步转向 Main V3 `marginal-gain teacher`。

## Main V2 计划

如果 richer candidates / marginal-gain labels 让 Main V1 出现正信号，再做 streaming memory selector：

```text
memory_{t-1}, x_t -> gain_t -> memory_t
```

评估方式：

- offline topK: 所有帧都到达后取 topK。
- streaming topK: 在线维护 size-K buffer。

## Main V3 计划

已完成第一版 frame-score 对照：

```bash
tmux new-session -d -s selector0005_m3_framescore '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v3_dinov2_vits14_frame_score_gated_m02 \
  --model-kind memory_frame_score \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 60 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --uniform-gate-margin 0.2 \
  --log-every-steps 10
'
```

结果：

- Val `uniform - learned = 0.0000`
- Test `uniform - learned = 0.0000`
- 结论：单纯 frame-score average 不足以超过 uniform。

下一步构建 greedy marginal-gain teacher：

- 对每个 scene 生成 candidate pool。
- 用 teacher score 估计每张图加入当前集合的 gain。
- 训练 student 预测 gain。

第一版不立刻补跑大规模 VGGT cache，而是复用 richer300 已有候选：

```text
candidate pool = uniform20 + uniform_jitter20_seed000-004 + random20_seed000-004
                 + contiguous20 + motion_spread20 + convnext_kcenter20 + dinov2_kcenter20
teacher utility = -target_error
student target = 在同一 scene 中，让高 utility candidates 覆盖的 frames 得到更高 marginal proxy
```

如果这个近似 teacher 能让 held-out test 稳定超过 `uniform20`，再进入 true marginal-gain cache：

```text
S_0 = empty or uniform prefix
for t in 1..K:
  sample M candidate additions i not in S
  run/cache VGGT for S + i
  label gain_i = utility(S + i) - utility(S)
```

已完成 ridge-gain 近似：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/merged_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v3_dinov2_vits14_frame_score_ridge_gain_w02 \
  --model-kind memory_frame_score \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 60 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --uniform-gate-margin 0.2 \
  --frame-target-mode ridge_gain \
  --frame-target-weight 0.2 \
  --frame-target-ridge 0.01 \
  --frame-target-clip 5.0 \
  --log-every-steps 10
```

对照：

| Run | Frame target weight | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---:|---:|---:|---|
| `main_v3_dinov2_vits14_frame_score_ridge_gain_w02` | `0.20` | `+0.0148` | `0.0000` | test 回 uniform |
| `main_v3_dinov2_vits14_frame_score_ridge_gain_w005` | `0.05` | `+0.0130` | `0.0000` | test 回 uniform |

结论：ridge 近似不能稳定超过 uniform；只有 candidate-set gated m=0.2 保留了 test `+0.0328` 的小正信号。若继续 Main V3，需要补跑 true step-gain cache，而不是继续调 frame-target weight。

## True Swap-Gain Labels

新增脚本：

```bash
scripts/run_stage2_image_only_swap_gain_labels.py
```

40-scene smoke：

```bash
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_swap_gain_labels.py \
  --run-dir runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_40 \
  --cache-root caches/vggt_omega/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_40_images512 \
  --limit-scenes 40 \
  --single-swaps 2 \
  --multi-swaps 2 \
  --cache-devices cuda:0,cuda:1 \
  --max-pixels-per-image 1024 \
  --max-pointmap-points 60000
```

300-scene 正式打标：

```bash
tmux new-session -d -s selector0005_swap300 '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_swap_gain_labels.py \
  --run-dir runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300 \
  --cache-root caches/vggt_omega/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300_images512 \
  --limit-scenes 300 \
  --single-swaps 2 \
  --multi-swaps 2 \
  --cache-devices cuda:0,cuda:1 \
  --max-pixels-per-image 1024 \
  --max-pointmap-points 60000 \
  2>&1 | tee runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/tmux.log
'
```

正式输出：

- Labels: `runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_hardlabel_train_labels.csv`
- Jobs: `runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_cache_jobs.json`
- Diagnostics: `runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/swap_gain_diagnostics.json`

关键诊断：

- `swap_best_win_rate_vs_uniform = 0.7300`
- `swap_oracle_rate = 0.2433`
- `swapgain20` oracle scenes: `73 / 300`

## Main V4 Swap-Gain 训练

基础训练命令：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v4_dinov2_swapgain300_candidate_set_gated_m02 \
  --model-kind memory_candidate_set \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 120 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --uniform-gate-margin 0.2 \
  --log-every-steps 10
```

Gate scan：

```bash
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/evaluate_stage2_image_only_selector_gate.py \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v4_dinov2_swapgain300_candidate_set_gated_m02 \
  --device cuda:0 \
  --batch-size 32 \
  --out runs/0005_image_only_teacher_student_selector/main_v4_dinov2_swapgain300_candidate_set_gated_m02/uniform_gate_scan.json
```

已完成对照：

| Run suffix | Extra args | 结果 |
|---|---|---|
| `gated_m02` | `--uniform-gate-margin 0.2` | val `+0.0095`, test `-0.3055` |
| `gated_m05` | `--uniform-gate-margin 0.5` | val `-0.0020`, test `-0.0087` |
| `gated_m10` | `--uniform-gate-margin 1.0` | val `+0.0359`, test `-0.0910` |
| `gated_m05_adv02` | `--uniform-gate-margin 0.5 --uniform-advantage-weight 0.2` | val `+0.0037`, test `-0.5211` |

`uniform_advantage_weight` 会让 candidate score 相对 `uniform20` 的 margin 拟合 `uniform20_error - candidate_error`。这版是负例，当前不建议继续沿这个 loss 调权重。

当前停止条件已满足：四个 full300 swap-gain run 的 val-selected gate 都没有在 held-out test 上超过 `uniform20`。下一步应换 objective，而不是继续重复同构训练。

## Main V5 Explicit Gate Head

新增脚本：

```bash
scripts/run_stage2_image_only_gate_head_training.py
```

目标：

- 复用 image-only DINOv2-S features 和 `memory_candidate_set` contextualizer。
- 对每个 candidate 显式预测 `advantage = uniform20_error - candidate_error`。
- 同时训练 binary `gate_logit` 判断 non-uniform candidate 是否值得偏离 `uniform20`。
- 推理时默认 `uniform20`，只有 gate/advantage 超过阈值才选择 non-uniform。

Smoke：

```bash
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_gate_head_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir runs/0005_image_only_teacher_student_selector/smoke_gate_head_20 \
  --limit-scenes 20 \
  --epochs 3 \
  --batch-size 4 \
  --hidden-dim 128 \
  --num-layers 1 \
  --num-heads 4 \
  --memory-slots 4 \
  --train-devices cuda:0 \
  --log-every-steps 1
```

正式 `advantage_weight=1.0`, `gate_weight=0.5`, `positive_margin=0.2`：

```bash
tmux new-session -d -s selector0005_gate_head_v5 '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
run_dir="runs/0005_image_only_teacher_student_selector/main_v5_dinov2_swapgain300_gate_head_aw1_gw05"
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_gate_head_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir "$run_dir" \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 120 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --advantage-weight 1.0 \
  --gate-weight 0.5 \
  --rank-weight 0.0 \
  --positive-margin 0.2 \
  --log-every-steps 20 \
  2>&1 | tee "$run_dir/tmux.log"
'
```

正式 `advantage_weight=0.5`, `gate_weight=1.0`, `positive_margin=0.5`：

```bash
tmux new-session -d -s selector0005_gate_head_v5_pm05 '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
run_dir="runs/0005_image_only_teacher_student_selector/main_v5_dinov2_swapgain300_gate_head_pm05_aw05_gw1"
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_gate_head_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir "$run_dir" \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 120 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0,cuda:1 \
  --advantage-weight 0.5 \
  --gate-weight 1.0 \
  --rank-weight 0.0 \
  --positive-margin 0.5 \
  --log-every-steps 20 \
  2>&1 | tee "$run_dir/tmux.log"
'
```

结果：

| Run suffix | Val-selected rule | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---|---:|---:|---|
| `aw1_gw05` | `gate_logit >= 1.0` | `+0.1120` | `-0.1174` | val 正，test 失败 |
| `pm05_aw05_gw1` | `gate_logit >= -1.0` | `+0.1303` | `-0.3096` | gate 过度偏离，test 更差 |

当前结论：显式 gate head 能增强 val 拟合，但没有解决 held-out test calibration。继续沿同一 300-scene split 调 loss 权重意义不大，应转向 larger/balanced split 或 patch/motion-aware student。

### Seed robustness

使用 `aw1_gw05` 设置补跑不同 seed。seed 会同时改变 train/val/test split 和模型初始化。

模板：

```bash
tmux new-session -d -s selector0005_gate_seed10 '
cd /home/m/project/ltm/selector &&
set -euo pipefail &&
run_dir="runs/0005_image_only_teacher_student_selector/main_v5_dinov2_swapgain300_gate_head_aw1_gw05_seed20260610"
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_gate_head_training.py \
  --labels-csv runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0005_image_only_teacher_student_selector/swap_gain_uniform20_dinov2_300/augmented_cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2_vits14 \
  --run-dir "$run_dir" \
  --seed 20260610 \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --epochs 120 \
  --batch-size 32 \
  --lr 2e-4 \
  --train-devices cuda:0 \
  --advantage-weight 1.0 \
  --gate-weight 0.5 \
  --rank-weight 0.0 \
  --positive-margin 0.2 \
  --log-every-steps 20 \
  2>&1 | tee "$run_dir/tmux.log"
'
```

完成 seed:

| Seed | Val-selected rule | Val `uniform - learned` | Test `uniform - learned` | Test oracle scan |
|---|---|---:|---:|---:|
| `20260609` | `gate_logit >= 1.0` | `+0.1120` | `-0.1174` | `+0.0000` |
| `20260610` | `advantage >= 0.05` | `+0.0811` | `+0.1163` | `+0.1485` |
| `20260611` | `gate_logit >= -0.5` | `+0.1341` | `-0.3548` | `+0.1399` |
| `20260612` | `advantage >= 0.3` | `+0.1180` | `-0.0763` | `+0.0000` |
| `20260613` | `advantage >= 0.0` | `+0.0969` | `-0.1539` | `+0.0108` |

汇总：

- Val mean `+0.1084`
- Test mean `-0.1172`
- Positive test seeds: `1 / 5`
- Test-oracle mean `+0.0599`

判断：gate head 有 split-dependent 正信号，但不能稳定泛化。下一步不应只继续加 seed，而应扩大 scene 数或升级 image-only feature 表示。

## 输出目录

建议：

```text
caches/image_features/0005/
runs/0005_image_only_teacher_student_selector/
docs/experiments/0005_image_only_teacher_student_selector/
```

## 记录要求

每个 run 至少记录：

- cheap backbone 名称与参数量。
- student 参数量。
- 是否使用任何 VGGT feature 作为 student input。
- val/test `uniform_minus_student_error`。
- inference cost。
- 与 `uniform20` 的 win rate。
