# Runbook

## Preconditions

- `0001_stage1_register_quality_gate` results are available.
- `0002_ltm30_pose_depth_validation` results are available and kept as held-out validation unless an explicit split update says otherwise.
- VGGT-OMEGA checkpoint and cache runner are available.
- `data/raw/ltm_datasets` points to `/home/m/dataset/ltm_datasets`.
- Train/val/test split is scene-level; subsets from one scene never cross splits.

## Planned Flow

0. Freeze the target definition.

   - Primary native metrics: `pose_rotation_mean_deg`, `pointmap_rmse_norm`, `depth_log_rmse`.
   - Secondary sanity metric: sensor `gt_depth_absrel_mean`.
   - Direct sensor `gt_pose_*` is excluded from the gate.

1. Build additional readout-calibration scenes.

   - Reuse the LTM preparation logic.
   - Prefer scenes with pose and depth.
   - Cap full images at 200 per scene.
   - Generate candidate subsets at `10%`, `20%`, and `30%`.
   - Include random, uniform, contiguous, k-center/farthest, and intentionally redundant negatives.
   - MVP source pool: WildRGBD + DL3DV, excluding the current LTM30 validation scenes.
   - Keep ScanNet pose-only scenes optional until the depth+pose readout baseline is validated.

2. Freeze scene splits.

   - `readout_train`: 300-500 scenes, no overlap with LTM30.
   - `readout_val`: current LTM30 plus optional extra held-out scenes.
   - `selector_train`: reserved for `0004`, preferably no scene overlap with `readout_train`.
   - `selector_val/test`: reserved for hard VGGT/FastGS/VLA validation.
   - Record split membership before VGGT cache generation.

3. Cache VGGT-OMEGA outputs.

   - full scene cache.
   - hard subset cache for every candidate subset.
   - required tensors: `camera_tokens.pt`, `register_tokens.pt`, `depth.pt`, `depth_conf.pt`, `pose_enc.pt`.

4. Compute labels.

   - subset-vs-full native depth consistency.
   - subset-vs-full pose rotation/center consistency.
   - derived point-map consistency.
   - optional sensor depth sanity metrics.

5. Train baselines.

   - parameter-free mean-pooled register cosine.
   - pooled MLP readout.
   - attention RegisterReadoutHead.

6. Validate by scene-held-out split.

   - Compute scene-wise Spearman/Pearson.
   - Compute expected-sign count.
   - Compute best-score vs best-quality match.
   - Compare every trained readout against mean pooling.

7. Decide.

   - If a readout passes gate, freeze checkpoint and promote it to `0004_stage2_fixed_k_selector_training`.
   - If none pass, keep mean pooling as the selector baseline objective and do not rely on a trained readout.

## Hard-Label Pilot Flow

After the `train500_full16` warmup, run `hardlabel100_full100_80`:

1. Select 50 WildRGBD and 50 DL3DV scenes, excluding LTM30 validation scenes.
2. Use full-view frames:

   - WildRGBD: 100 frames.
   - DL3DV: 80 frames.

3. Generate 12 hard subsets per scene:

   - 5 x `random20`.
   - 3 x `random50`.
   - `uniform20`, `uniform50`.
   - `contiguous20_seed000`, `contiguous50_seed000`.

4. Cache VGGT-OMEGA with depth and pose for full and all hard subsets.
5. Compute subset-vs-full native geometry labels.
6. Train pooled readout with pairwise ranking over same-scene good/bad subsets.
7. Validate on LTM30 hard subset metrics.
8. If pooled readout still fails, keep the hard labels and train the 2-layer attention readout.

Implementation command:

```bash
/home/m/project/ltm/vggt-omega/.venv/bin/python scripts/prepare_stage2_readout_hardlabel100.py
```

## Attention Multi-Metric Follow-Up

Reuse the completed hard-label cache and labels from `hardlabel100_pooled_mlp_full100_80`; do not rerun VGGT cache.

Target command:

```bash
tmux new -s readout0003_attention_multimetric
/home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_readout_attention_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv \
  --run-dir runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric \
  --train-devices cuda:0,cuda:1 \
  --epochs 30 \
  --batch-size 16 \
  --pairs-per-scene-metric 24 \
  --num-workers 4
```

Primary evaluation is metric-head LTM30 expected alignment. Embedding-cosine expected alignment is reported as a secondary diagnostic.

Completed result: best metric-head expected alignment reached `0.5657` at epoch 25, which is slightly above pooled hard-label readout `0.5594` but below the strict promotion gate.

```bash
tmux new -s readout0003_hardlabel100
/home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_readout_hardlabel_training.py \
  --manifest docs/experiments/0003_stage2_readout_calibration/hardlabel100_manifest.json \
  --cache-root caches/vggt_omega/0003_stage2_readout_calibration/hardlabel100_full100_80_images512 \
  --run-dir runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80 \
  --cache-devices cuda:0,cuda:1 \
  --train-devices cuda:0,cuda:1 \
  --epochs 40 \
  --batch-size 12 \
  --pairs-per-scene 48
```

## Future Commands

These are target command shapes for the implementation phase; they should not be run until the corresponding CLI/config exists.

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage readout-build-dataset \
  --config configs/experiments/0003_stage2_readout_calibration.yaml
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage readout-cache-vggt \
  --config configs/experiments/0003_stage2_readout_calibration.yaml
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage readout-train \
  --config configs/experiments/0003_stage2_readout_calibration.yaml
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage readout-eval \
  --config configs/experiments/0003_stage2_readout_calibration.yaml \
  --checkpoint <readout_checkpoint>
```

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage record-run \
  --experiment 0003_stage2_readout_calibration \
  --stage stage2 \
  --method attention_readout \
  --dataset <scene_split_name> \
  --config configs/experiments/0003_stage2_readout_calibration.yaml \
  --notes "scene-held-out readout calibration"
```

## Required Artifacts

- scene split manifest.
- subset manifest with generation method, ratio, seed, selected image list.
- VGGT-OMEGA cache manifest for full and subset runs.
- native geometry labels per subset.
- readout checkpoints.
- training curves.
- evaluation CSVs for correlation, sign, and best-match metrics.
- frozen readout promotion note for `0004`, if gate passes.

## Checks

- LTM30 validation scenes are not used for readout training unless the split document explicitly changes.
- Selector training/evaluation scenes for `0004` are not silently reused from readout validation/test.
- `R` register count and `C` token dim are read from cache manifest/tensor shape, not hard-coded.
- All correlations are computed inside each scene first, then summarized across scenes.
- Lower-is-better native errors are evaluated with expected negative correlation to readout score/similarity.
- Direct sensor pose metrics are reported only as diagnostics, not gate metrics.
- A readout checkpoint is not used by `0004` unless its exact checkpoint path and validation summary are recorded.
