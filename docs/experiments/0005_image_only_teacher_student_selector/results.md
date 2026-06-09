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

## Richer Candidates 进度

为解决 hardlabel300 candidate pool 过于偏向 `uniform20` 的问题，已实现 richer-candidate label 生成流程，并完成 4-scene smoke。

新增候选：

- `uniform_jitter20_seed000-004`
- `convnext_kcenter20_seed000`
- `dinov2_kcenter20_seed000`
- `motion_spread20_seed000`

4-scene smoke:

- Run dir: `runs/0005_image_only_teacher_student_selector/richer_candidates_smoke4`
- Cache root: `caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_smoke4_images512`
- Jobs: `36 = 4 scenes * (1 full + 8 subset)`
- Result: cache + labels 成功。
- Smoke merged labels: `2132` rows。
- Smoke oracle 新增 `uniform_jitter20 = 4`，说明 richer candidates 确实能制造比旧候选更有信息量的监督。

正式 hardlabel300 richer-candidate cache:

- Run dir: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300`
- Cache root: `caches/vggt_omega/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_images512`
- Total jobs: `2700 = 300 scenes * (1 full + 8 subset)`
- 当前完成: `1706 / 2700`
- 当前缺失: `994 / 2700`
- 完整 scene: `108 / 300` 已具备 `full + 8 subset` 全部 cache。
- Missing jobs list: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300/missing_cache_jobs_after_gpu_failure.json`
- 当前 cache 占用: 约 `105G`

阻塞原因：

- 正式 cache 过程中出现 `CUDA error: unspecified launch failure`。
- 随后 `nvidia-smi` 只枚举到一张 RTX 5090，单卡 resume 又在 CUDA 初始化阶段报 `CUDA unknown error`。
- 因此 richer-candidate 正式 labels 尚未完成，暂不能启动最终 richer-candidate selector training。

## Ready108 Partial 诊断

在 GPU 恢复前，使用 `--ready-only` 对 108 个已完整缓存的 scene 先做 partial label 诊断：

- Run dir: `runs/0005_image_only_teacher_student_selector/richer_candidates_hardlabel300_ready108`
- Labels: `1620 = 108 scenes * (7 old candidates + 8 richer candidates)`
- 数据分布: `DL3DV-ALL-480P = 57`, `wildrgbd_harrison = 51`
- 诊断输出: `partial_richer108_diagnostic_summary.json`

Oracle family:

| Family | Oracle scenes |
|---|---:|
| `uniform_jitter20` | 76 |
| `uniform20` | 14 |
| `random20` | 8 |
| `dinov2_kcenter20` | 5 |
| `convnext_kcenter20` | 5 |

关键判断：

- `uniform20` 不再垄断 oracle，从旧池的 `217/300` 强势状态降为 ready108 的 `14/108`。
- richer-best 比 old-best 平均低 `0.4439` target_error。
- richer candidates 在 `86/108` 个 scene 上优于旧候选池 best。
- `uniform_minus_oracle_error = 0.9060`，说明这批 richer labels 有足够 teacher margin。

## Ready108 CPU 训练诊断

由于当前 CUDA 不可用，先用 CPU 跑小型 `memory_candidate_set` 诊断，只作为信号检查，不作为 promotion 结论。

| Run | Backbone | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---|---:|---:|---|
| `main_v1_convnext_tiny_richer108_cpu` | ConvNeXt-Tiny | `+0.0814` | `-0.2163` | val 有信号，但 test 不稳 |
| `main_v1_dinov2_vits14_richer108_cpu` | DINOv2-S/ViT-S | `+0.2447` | `+0.0436` | partial 正信号，值得补全 300-scene |

DINOv2-S ready108 best checkpoint:

- Train: `uniform_minus_learned_error = +0.5351`
- Val: `+0.2447`, `win_rate_vs_uniform = 0.2727`, `pairwise_accuracy = 0.7076`
- Test: `+0.0436`, `win_rate_vs_uniform = 0.2000`, `pairwise_accuracy = 0.7483`
- Test learned methods: `uniform20 = 5`, `uniform_jitter20 = 5`

这个结果说明：一旦 candidate pool 不再过度偏向 `uniform20`，image-only student 至少可以在 partial set 上学到一点正向 deviation。正式结论仍需等 GPU 恢复后补全 `300` scene richer cache，再做双卡正式训练。

## 记录口径

只有当 student 推理时不读取 VGGT-OMEGA tokens/features，结果才计入 0005。

如果某个 run 使用 VGGT-OMEGA per-frame tokens 作为输入，应归入 0004 或另记为 teacher/diagnostic ablation，不能作为 image-only selector 结果。
