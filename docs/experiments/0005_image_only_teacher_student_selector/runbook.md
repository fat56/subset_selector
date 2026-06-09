# 运行手册

## 当前状态

`0005` 目前是 design-draft，尚未实现代码。第一步不是训练，而是把 teacher/student 数据流和 evaluation gate 固定下来，避免再次落回“推理时依赖 full VGGT features”的 0004 路线。

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

## 预期脚本

尚未创建，建议后续按下面名字实现：

```bash
scripts/prepare_stage2_image_only_selector_features.py
scripts/run_stage2_image_only_selector_training.py
scripts/evaluate_stage2_image_only_selector.py
```

## 推荐 smoke

第一版 smoke 只用 hardlabel300：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/prepare_stage2_image_only_selector_features.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --out-dir caches/image_features/0005/hardlabel300_dinov2s \
  --backbone dinov2_vits14 \
  --device cuda:0
```

训练命令建议：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_selector_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --feature-cache caches/image_features/0005/hardlabel300_dinov2s \
  --run-dir runs/0005_image_only_teacher_student_selector/main_v1_dinov2_candidate_rank \
  --model-kind memory_candidate_set \
  --memory-slots 8 \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 60 \
  --batch-size 32 \
  --train-devices cuda:0,cuda:1
```

## Main V1 判定

通过条件：

- val `uniform_minus_student_error > 0`。
- held-out test `uniform_minus_student_error > 0`。
- student inference 不读取 VGGT features。

不通过条件：

- val 有提升但 test 为负，视为 calibration 失败。
- test oracle scan 最优为 uniform fallback，说明当前 student 不可靠。

## Main V2 计划

如果 Main V1 通过，再做 streaming memory selector：

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
