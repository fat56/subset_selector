# Runbook

## Preconditions

- Free disk on `/` is at least `650G`; current check showed about `814G`.
- `runs/0006_stage2_step_gain_teacher/...` labels are preserved for comparison.
- `caches/image_features/` is preserved.
- LTM30 validation scenes remain excluded from training scenes.

## Step 0: Space Check

```bash
df -h /home/m/project/ltm/selector /home/m
du -sh caches runs caches/vggt_omega caches/image_features 2>/dev/null
```

Stop before label generation if free space is below `650G`.

## Step 1: Build 1000-Scene Manifest

The current `0006` label script is tied to hardlabel300/richer300 inputs. First create a manifest or label source that covers about 1000 scenes.

Target manifest:

```text
docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_manifest.json
docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_scenes.csv
docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_summary.md
```

Preferred source mix:

```text
500 WildRGBD Harrison scenes
500 DL3DV scenes
```

If reusing the hard-label manifest builder, use a new stem and do not overwrite `0003` artifacts.

Expected command shape:

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

Review the output before launching VGGT:

```bash
sed -n '1,120p' docs/experiments/0007_stage2_swap_gain_scaleup/swapgain1000_summary.md
```

## Step 2: Prepare Image-Only Features

The direct gain regressor needs one image-only feature file per scene. Use DINOv2-S/ViT-S first.

If `prepare_stage2_image_only_selector_features.py` still requires labels/jobs rather than a manifest, add a manifest-based mode before running the full cache. Do not generate VGGT labels just to create image-only features.

Expected output:

```text
caches/image_features/0007/swapgain1000_dinov2_vits14/<scene_id>.pt
```

Optional secondary feature cache:

```text
caches/image_features/0007/swapgain1000_dinov2_patch_summary_temporal/<scene_id>.pt
```

## Step 3: Smoke Label Generation

Run a small smoke before the 1000-scene cache.

Target:

- `20` scenes.
- `8` single swaps.
- Both datasets represented.

Expected output:

```text
runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8_smoke20/
caches/vggt_omega/0007_stage2_swap_gain_scaleup/swapgain1000_single8_smoke20_images512/
```

Checks:

- All VGGT jobs complete.
- `augmented_hardlabel_train_labels.csv` exists.
- Teacher best swap beats `uniform20` on most smoke scenes.
- Per-scene cache cost is close to the `0006` estimate.

## Step 4: Full 1000 x 8 Label Generation

Launch only after the smoke passes.

Target paths:

```text
runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8/
caches/vggt_omega/0007_stage2_swap_gain_scaleup/swapgain1000_single8_images512/
```

Expected scale:

- `1000 scenes`.
- `8000` single-swap VGGT jobs.
- About `340G` new swap cache, plus any full/uniform reference cache not reused.

Monitor:

```bash
df -h /home/m
du -sh caches/vggt_omega/0007_stage2_swap_gain_scaleup/* 2>/dev/null
```

Stop condition:

- Free disk below `250G`.
- Repeated VGGT cache failures on the same device.
- Per-scene cache cost exceeds the `0006` estimate by more than `30%`.

## Step 5: Direct Gain Regressor, Five Seeds

Primary training command shape:

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

Repeat for seeds:

```text
20260619
20260620
20260621
20260622
20260623
```

Use one GPU per run if running seeds in parallel. Keep run directories separate.

## Step 6: Summarize

Update `results.md` with:

- Teacher oracle diagnostics.
- Per-seed val/test deltas.
- Mean/median/worst test delta.
- Positive seed count.
- Dataset-wise WildRGBD vs DL3DV deltas.
- Gain MAE/sign/pairwise accuracy.
- Disk usage after label generation.

Promotion requires:

```text
mean test delta >= +0.05
median test delta > 0
positive seeds >= 4/5
worst seed >= -0.02
```

## Cleanup Policy

After labels and summaries are written, VGGT tensor cache may be deleted if disk pressure returns, but preserve:

```text
runs/0007_stage2_swap_gain_scaleup/**/augmented_hardlabel_train_labels.csv
runs/0007_stage2_swap_gain_scaleup/**/augmented_cache_jobs.json
runs/0007_stage2_swap_gain_scaleup/**/summary.json
runs/0007_stage2_swap_gain_scaleup/**/gate_scan.json
caches/image_features/0007/
```
