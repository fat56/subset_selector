# 复盘

## 当前判断

0005 值得单独开线，但 `Main V1 hardlabel300 candidate-rank` 第一轮没有得到可推广提升。

原因是 0004 的核心输入是 full scene 的 VGGT-OMEGA compact features，推理阶段仍需要先跑 VGGT，因此不符合“先选子集，再运行 VGGT/3DGS”的原始目标。

0005 的目标更明确：用 VGGT-OMEGA teacher 生成监督，但训练一个推理时只看原始图像或 cheap image features 的 student selector。

## 已完成的 Main V1

本轮实现并运行了两个 image-only backbone：

- `ConvNeXt-Tiny`
- `DINOv2-S/ViT-S`

网络是 `memory_candidate_set`：

```text
image_i
  -> frozen cheap backbone
  -> image-only frame feature
  -> projection MLP
  -> latent memory slots
  -> candidate mask pooling
  -> candidate score
```

训练 label 复用 `0003` hardlabel300 的 hard-native `target_error`，只在已有 candidates 中做 ranking：

- `uniform20`
- `random20_seed000-004`
- `contiguous20_seed000`

student 输入没有 VGGT-OMEGA `camera_token`、`register_tokens` 或 full VGGT output。

## 结果判断

两个 run 的 best-val checkpoint 都退回 `uniform20`：

| Run | Val uniform - student | Test uniform - student | 判断 |
|---|---:|---:|---|
| `main_v1_convnext_tiny_candidate_rank` | `0.0000` | `0.0000` | 未超过 uniform |
| `main_v1_dinov2_vits14_candidate_rank` | `0.0000` | `0.0000` | 未超过 uniform |

这不是代码失败，而是一个很有用的诊断：

- candidate pool 里 `uniform20` 是 `217/300` 个 scene 的 oracle。
- 偏离 uniform 的 epoch 在 val 上都变差。
- pairwise accuracy 约 `0.69-0.72`，说明模型能读到部分排序信号，但这个信号不足以支撑可靠的 top candidate deviation。
- DINOv2-S 没有明显优于 ConvNeXt-Tiny，至少在 global embedding + hardlabel300 candidate-rank 这个设定下不是瓶颈。

## 建议路线

第一步已经回答了一个基础问题：

> cheap image features 是否足以预测 hard-native candidate ranking？

在当前 hardlabel300 candidate pool 下，答案是：不足以超过 `uniform20`。但这个结论不等价于 image-only selector 没希望，因为当前 label 设计过于偏向 uniform，且 candidate family 太窄。

下一步不建议继续只换 backbone 或堆训练轮次。更值得推进的是改 teacher label/candidate 设计：

- 增加 `uniform_jitter20`、`cheap_feature_kcenter20`、`dino_kcenter20`、`motion_spread20` 等更有竞争力的 candidates。
- 把 objective 从“候选 top1 分类”改为 margin-aware gate：只有当 teacher 明确优于 uniform 时才允许 deviation。
- 构建 `Main V3 marginal-gain teacher`，让 student 学“加入当前集合后的增益”，而不是在偏 uniform 的候选池里猜 top1。
- 如果仍用 Main V1，应该先扩到 hardlabel1000 + richer candidates，而不是在 hardlabel300 上继续加 epoch。

## Richer Candidates 更新

已按上述判断实现 richer-candidate label 流程，并完成两层诊断：

1. `4-scene smoke` 成功，证明新增候选与 native metric 计算链路可跑通。
2. 正式 hardlabel300 cache 跑到 `1706/2700` jobs 后被 GPU/driver 状态阻塞；其中 `108/300` scene 已完整拥有 `full + 8 subset` cache。

对这 `108` 个完整 scene 做 `ready-only` partial labels 后，oracle 分布明显改善：

| Family | Oracle scenes |
|---|---:|
| `uniform_jitter20` | 76 |
| `uniform20` | 14 |
| `random20` | 8 |
| `dinov2_kcenter20` | 5 |
| `convnext_kcenter20` | 5 |

这说明第一轮失败的主要原因确实是旧 candidate pool 过窄、过度偏向 `uniform20`。在 ready108 上，richer-best 比 old-best 平均低 `0.4439` target_error，并且在 `86/108` 个 scene 上优于旧候选池 best。

## Ready108 Student 诊断

CUDA 当前不可用，因此先用 CPU 跑了小模型诊断：

| Run | Val `uniform - learned` | Test `uniform - learned` | 判断 |
|---|---:|---:|---|
| ConvNeXt-Tiny ready108 | `+0.0814` | `-0.2163` | 不稳 |
| DINOv2-S ready108 | `+0.2447` | `+0.0436` | 有 partial 正信号 |

DINOv2-S 的 test 提升很小，不能直接 promotion；但它已经不同于 v0 的 `0.0000` fallback，说明 richer labels 让 image-only student 开始学到可用 deviation。

当前最合理的判断是：

- 继续补全 hardlabel300 richer cache，而不是回到旧 label 上调参。
- 补全后优先跑 DINOv2-S richer300 正式训练；ConvNeXt-Tiny 作为轻量对照。
- 如果 richer300 仍只给很小提升，再进入 Main V3 marginal-gain teacher，而不是继续堆 `memory_candidate_set` 容量。

## 当前风险

- `uniform20` 是很强 baseline，本轮已确认 student 偏离后 val 变差。
- hardlabel300 样本太少，仍可能出现 0004 的 val/test 反转。
- cheap image backbone 的语义特征未必等价于三维 coverage。
- global embedding 可能丢失视角覆盖信息，后续可能需要 patch summary / TokenLearner tokens / optical-flow-like motion proxy。

## 下一步

- 不把本轮 checkpoint 推进到 hard subset VGGT rerun。
- 保留 ConvNeXt-Tiny 和 DINOv2-S feature cache 作为后续 ablation 资产。
- GPU/driver 恢复后，先续跑缺失的 `994` 个 richer cache jobs。
- 缓存补全后生成正式 hardlabel300 richer labels，并启动 DINOv2-S / ConvNeXt-Tiny richer300 双卡训练。
