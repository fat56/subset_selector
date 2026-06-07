# Runbook

## Preconditions

- `0003_stage2_readout_calibration` 已明确采用 frozen readout checkpoint 或 mean-pooling baseline objective。
- VGGT-OMEGA checkpoint、FastGS/3DGS backend、dataset registry 均已可用。
- 每个 scene 按 train/val/test 做场景级划分，同一 scene 不跨 split。
- full scene 的 VGGT-OMEGA camera/register token cache 已准备好，或已有生成计划。
- 本实验设计已被接受后，才创建 `configs/experiments/0004_stage2_fixed_k_selector_training.yaml` 和对应 `src` 实现。

## Planned Flow

0. Freeze the design.

   - 确认第一版 `K`：推荐沿用 Stage 1 的 `20%` ratio。
   - 确认 selector 输入：推荐使用 cached VGGT camera/register summaries。
   - 确认 readout 状态：使用 `0003` 选定并冻结的 readout；如果 `0003` 未通过，则使用 mean pooling baseline objective。

1. Build caches.

   - `z_full` for every train/val/test scene。
   - per-image `camera_token_i`。
   - per-image `register_mean_i` and `register_max_i`。
   - optional DINO/CLIP/image-quality/pose/overlap features。

2. Import readout/proxy decision.

   从 `0003_stage2_readout_calibration` 导入 frozen readout checkpoint、validation summary、或 mean-pooling fallback decision。本实验不重新训练 readout。

3. Train soft-token selector.

   - input: cached per-image features。
   - model: FeatureProjector + 4-layer SetSelector + ScoreHead。
   - selection: relaxed top-K mask。
   - MVP objective: `L_pos`, plus optional `L_nce` when batch scenes are diverse enough。
   - auxiliary losses such as coverage, redundancy, quality, depth, or pose stay off until a concrete failure mode appears。

4. Run hard-subset validation.

   - 对 val scenes 用 `topK(scores, K)` 得到 selected indices。
   - 按原始帧序排序 selected indices。
   - 重新运行 frozen VGGT-OMEGA on selected images。
   - 计算 hard `register_cosine_similarity`。

5. Run FastGS/3DGS validation.

   - learned selector 与 Stage 1 baselines 使用完全相同的 scene、K、backend 设置。
   - 记录 PSNR/SSIM/LPIPS、运行时间和失败 case。

6. Decide.

   - 通过则进入 Stage 3 variable-K/Pareto 设计。
   - 不通过则根据 soft/hard gap 或 FastGS gap 选择 ranking refinement、coverage/geometry auxiliary，或停止。

## Future Commands

以下命令是实现后的目标形态，不应在当前文档阶段直接运行。

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage stage2-cache \
  --config configs/experiments/0004_stage2_fixed_k_selector_training.yaml \
  --dataset <dataset_or_scene_set>
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage stage2-train \
  --config configs/experiments/0004_stage2_fixed_k_selector_training.yaml \
  --dataset <dataset_or_scene_set>
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage stage2-eval-hard \
  --config configs/experiments/0004_stage2_fixed_k_selector_training.yaml \
  --checkpoint <selector_checkpoint> \
  --dataset <dataset_or_scene_set>
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage record-run \
  --experiment 0004_stage2_fixed_k_selector_training \
  --stage stage2 \
  --method learned_fixed_k_selector \
  --dataset <dataset_or_scene_set> \
  --config configs/experiments/0004_stage2_fixed_k_selector_training.yaml \
  --notes "fixed K selector hard validation"
```

## Required Artifacts

- `z_full.pt` or equivalent full-scene embedding cache。
- `per_image_features.pt` or per-scene feature shards。
- optional readout checkpoint and calibration manifest。
- selector checkpoint。
- `selected_indices.json` for every evaluated scene。
- `stage2_subset_manifest.json`。
- hard VGGT-OMEGA cache for selected subsets。
- FastGS/3DGS metrics JSON。
- training curves: loss, soft cosine, hard cosine, retrieval accuracy。
- `manifest.yaml` with config path, checkpoint path, dataset split, K, and backend version。

## Checks

- readout checkpoint is fixed during selector training unless an explicit ablation says otherwise。
- `R` register count is read from VGGT-OMEGA checkpoint/config, not hard-coded。
- train/val/test split is scene-level。
- hard selected indices are sorted by original frame order before VGGT-OMEGA。
- learned selector is compared against the exact Stage 1 baselines at the same `K`。
- final decision uses hard-subset FastGS/3DGS metrics, not only soft proxy loss。
