# 结果

## 当前状态

`Main V1` 已完成第一轮 image-only hardlabel300 candidate-rank 实验：

- `main_v1_convnext_tiny_candidate_rank`
- `main_v1_dinov2_vits14_candidate_rank`

这两个 run 都严格遵守 0005 边界：student 输入是原始图像经过 cheap backbone 得到的 image-only features，训练/推理均不读取 VGGT-OMEGA tokens/features。

## 指标

| Run | Student input | Teacher labels | Val uniform - student | Test uniform - student | 结论 |
|---|---|---|---:|---:|---|
| `main_v1_convnext_tiny_candidate_rank` | ConvNeXt-Tiny global embedding + image stats | hardlabel300 | `0.0000` | `0.0000` | best-val checkpoint 退回 `uniform20` |
| `main_v1_dinov2_vits14_candidate_rank` | DINOv2-S/ViT-S global embedding + image stats | hardlabel300 | `0.0000` | `0.0000` | best-val checkpoint 退回 `uniform20` |

解释：

- `uniform - student > 0` 才表示 student 比 `uniform20` 更好。
- 两个 backbone 的 best-val checkpoint 都没有得到正提升。
- 训练中存在偏离 `uniform20` 的 epoch，但 val `uniform - student` 都为负，说明偏离 uniform 会变差。

## 详细结果

### `main_v1_convnext_tiny_candidate_rank`

- Feature cache: `caches/image_features/0005/hardlabel300_convnext_tiny`
- Run dir: `runs/0005_image_only_teacher_student_selector/main_v1_convnext_tiny_candidate_rank`
- Feature cache 耗时: `125.5s`
- Training 耗时: `26.17s`
- Feature cache 大小: `44.3 MiB`
- Backbone feature dim: `768`

Best-val checkpoint:

| Split | Learned mean error | Uniform20 mean error | Uniform - learned | Oracle top1 | Pairwise acc |
|---|---:|---:|---:|---:|---:|
| train | `-0.8582` | `-0.8582` | `0.0000` | `0.7167` | `0.7074` |
| val | `-1.0574` | `-1.0574` | `0.0000` | `0.7667` | `0.7186` |
| test | `-0.9917` | `-0.9917` | `0.0000` | `0.7333` | `0.6994` |

### `main_v1_dinov2_vits14_candidate_rank`

- Feature cache: `caches/image_features/0005/hardlabel300_dinov2_vits14`
- Run dir: `runs/0005_image_only_teacher_student_selector/main_v1_dinov2_vits14_candidate_rank`
- Feature cache 耗时: `141.8s`
- Training 耗时: `23.25s`
- Feature cache 大小: `24.6 MiB`
- Backbone feature dim: `384`

Best-val checkpoint:

| Split | Learned mean error | Uniform20 mean error | Uniform - learned | Oracle top1 | Pairwise acc |
|---|---:|---:|---:|---:|---:|
| train | `-0.8582` | `-0.8582` | `0.0000` | `0.7167` | `0.6992` |
| val | `-1.0574` | `-1.0574` | `0.0000` | `0.7667` | `0.6884` |
| test | `-0.9917` | `-0.9917` | `0.0000` | `0.7333` | `0.7251` |

## Dataset 诊断

本轮 hardlabel300 candidate pool:

- scenes: `300`
- split: train `240`, val `30`, test `30`
- datasets: DL3DV `150`, WildRGBD `150`
- candidates per scene: `uniform20`, `random20_seed000-004`, `contiguous20_seed000`
- oracle counts: `uniform20 = 217`, `random20 = 82`, `contiguous20 = 1`

这说明当前 labeled candidates 中 `uniform20` 本身非常强，占 oracle 的 `72.3%`。因此 Main V1 candidate-rank 模型最稳的 validation 策略就是选择 `uniform20`，而不是学习非 uniform deviation。

## 记录口径

只有当 student 推理时不读取 VGGT-OMEGA tokens/features，结果才计入 0005。

如果某个 run 使用 VGGT-OMEGA per-frame tokens 作为输入，应归入 0004 或另记为 teacher/diagnostic ablation，不能作为 image-only selector 结果。
