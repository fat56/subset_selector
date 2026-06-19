# 运行手册

## 前置条件

- `/` 上至少有 `650G` 可用空间；当时检查约为 `814G`。
- `runs/0006_stage2_step_gain_teacher/...` 标签保留，用于对比。
- `caches/image_features/` 保留。
- LTM30 validation scenes 继续从训练场景中排除。

## 步骤 0: 空间检查

```bash
df -h /home/m/project/ltm/selector /home/m
du -sh caches runs caches/vggt_omega caches/image_features 2>/dev/null
```

如果 label generation 前可用空间低于 `650G`，停止启动。

## 步骤 1: 构建 1000 场景 Manifest

当前 `0006` label script 绑定了 hardlabel300/richer300 输入。需要先创建覆盖约 1000 个场景的 manifest 或 label source。

目标 manifest：

```text
docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_manifest.json
docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_scenes.csv
docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_summary.md
```

优先数据配比：

```text
500 WildRGBD Harrison scenes
500 DL3DV scenes
```

如果复用 hard-label manifest builder，使用新的 stem，不要覆盖 `0003` 产物。

预期命令形态：

```bash
PYTHONPATH=scripts:src python scripts/prepare_stage2_readout_hardlabel100.py \
  --out-dir docs/experiments/0007_stage2_swap_gain_scaleup \
  --manifest-stem swapgain1000 \
  --name swapgain1000_full100_80 \
  --wildrgbd-scenes 500 \
  --dl3dv-scenes 500 \
  --wildrgbd-full-frames 100 \
  --dl3dv-full-frames 80 \
  --random-seed 20260619
```

启动 VGGT 前检查输出：

```bash
sed -n '1,120p' docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_summary.md
```

## 步骤 2: 准备 Image-Only Features

Direct gain regressor 需要每个场景一个 image-only feature 文件。第一版使用 DINOv2-S/ViT-S。

如果 `prepare_stage2_image_only_selector_features.py` 仍要求 labels/jobs，而不是 manifest，则先添加 manifest-based mode，再运行 full feature cache。不要为了创建 image-only features 去生成 VGGT labels。

预期输出：

```text
caches/image_features/0007/swapgain1000_dinov2_vits14/<scene_id>.pt
```

可选二级 feature cache：

```text
caches/image_features/0007/swapgain1000_dinov2_patch_summary_temporal/<scene_id>.pt
```

## 步骤 3: Smoke 标签生成

在 1000 场景 cache 前先跑小规模 smoke。

目标：

- `20` 个场景。
- `8` 个 single swaps。
- 两个数据集都要覆盖。

预期输出：

```text
runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8_smoke20/
caches/vggt_omega/0007_stage2_swap_gain_scaleup/swapgain1000_single8_smoke20_images512/
```

检查项：

- 所有 VGGT jobs 完成。
- `augmented_hardlabel_train_labels.csv` 存在。
- Teacher best swap 在大多数 smoke scenes 上优于 `uniform20`。
- 单场景 cache 成本接近 `0006` 估计。

## 步骤 4: Full 1000 x 8 标签生成

仅在 smoke 通过后启动。

目标路径：

```text
runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8/
caches/vggt_omega/0007_stage2_swap_gain_scaleup/swapgain1000_single8_images512/
```

预期规模：

- `1000 scenes`。
- `8000` 个 single-swap VGGT jobs。
- 约 `340G` 新 swap cache，外加无法复用的 full/uniform reference cache。

实际规模：

- `1000` 个场景。
- 包含 `uniform20` 在内，共 `9000` 行标签。
- Full VGGT cache 实际为 `596G`，路径为 `caches/vggt_omega/0007_stage2_swap_gain_scaleup/swapgain1000_single8_images512`。
- 标签和训练完成后剩余空间为 `208G`。

监控：

```bash
df -h /home/m
du -sh caches/vggt_omega/0007_stage2_swap_gain_scaleup/* 2>/dev/null
```

停止条件：

- 可用空间低于 `250G`。
- 同一设备上反复出现 VGGT cache failure。
- 单场景 cache 成本超过 `0006` 估计 `30%` 以上。

## 步骤 5: Direct Gain Regressor，五个 Seed

主训练命令形态：

```bash
PYTHONPATH=scripts:src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_image_only_swap_gain_regressor.py \
  --labels-csv runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8/augmented_hardlabel_train_labels.csv \
  --cache-jobs-json runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8/augmented_cache_jobs.json \
  --feature-cache caches/image_features/0007/swapgain1000_dinov2_vits14 \
  --run-dir runs/0007_stage2_swap_gain_scaleup/swap_gain_regressor_global_dino_seed20260619 \
  --candidate-tag 20 \
  --seed 20260619 \
  --epochs 120 \
  --batch-size 32 \
  --lr 2e-4 \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 8 \
  --memory-slots 8 \
  --dropout 0.1 \
  --regression-weight 1.0 \
  --sign-weight 0.3 \
  --rank-weight 0.3 \
  --gain-clip 4.0 \
  --train-devices cuda:0 \
  --log-every-steps 20
```

重复以下 seeds：

```text
20260619
20260620
20260621
20260622
20260623
```

如果并行跑多个 seed，每个 run 使用一张 GPU。不同 seed 的 run directory 必须分开。

## 步骤 6: 总结

更新 `results.md`，包含：

- Teacher oracle diagnostics。
- 每个 seed 的 val/test deltas。
- Mean/median/worst test delta。
- Positive seed count。
- WildRGBD vs DL3DV 的 dataset-wise delta。
- Gain MAE/sign/pairwise accuracy。
- Label generation 后的磁盘使用情况。

晋级要求：

```text
mean test delta >= +0.05
median test delta > 0
positive seeds >= 4/5
worst seed >= -0.02
```

## 清理策略

Labels 和 summaries 已写出后，如果磁盘压力回升，可以删除 VGGT tensor cache，但需要保留：

```text
runs/0007_stage2_swap_gain_scaleup/**/augmented_hardlabel_train_labels.csv
runs/0007_stage2_swap_gain_scaleup/**/augmented_cache_jobs.json
runs/0007_stage2_swap_gain_scaleup/**/summary.json
runs/0007_stage2_swap_gain_scaleup/**/gate_scan.json
caches/image_features/0007/
```
