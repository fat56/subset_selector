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

## Main V2 计划

如果 richer candidates / marginal-gain labels 让 Main V1 出现正信号，再做 streaming memory selector：

```text
memory_{t-1}, x_t -> gain_t -> memory_t
```

评估方式：

- offline topK: 所有帧都到达后取 topK。
- streaming topK: 在线维护 size-K buffer。

## Main V3 计划

如果 streaming selector 有希望，再构建 greedy marginal-gain teacher：

- 对每个 scene 生成 candidate pool。
- 用 teacher score 估计每张图加入当前集合的 gain。
- 训练 student 预测 gain。

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
