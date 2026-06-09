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

## 当前风险

- `uniform20` 是很强 baseline，本轮已确认 student 偏离后 val 变差。
- hardlabel300 样本太少，仍可能出现 0004 的 val/test 反转。
- cheap image backbone 的语义特征未必等价于三维 coverage。
- global embedding 可能丢失视角覆盖信息，后续可能需要 patch summary / TokenLearner tokens / optical-flow-like motion proxy。

## 下一步

- 不把本轮 checkpoint 推进到 hard subset VGGT rerun。
- 保留 ConvNeXt-Tiny 和 DINOv2-S feature cache 作为后续 ablation 资产。
- 优先设计 richer candidate labels 或 `marginal-gain` teacher，再训练下一版 0005。
