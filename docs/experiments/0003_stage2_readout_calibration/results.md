# Results

## Run: `train500_pooled_mlp_full16`

- Date: 2026-06-07
- Train scenes: 500
- Train source: 250 WildRGBD + 250 DL3DV
- Full-view frames per scene: 16
- Cached training images: 8,000
- Cache devices: `cuda:0,cuda:1`
- Train device: `cuda:0`
- Model: pooled MLP readout
- Objective: full-token online subset-mask warmup with positive cosine, symmetric InfoNCE, and uniform-vs-contiguous ranking.
- Validation: LTM30 hard subset validation, 30 scenes x 6 subsets = 180 rows.
- Run dir: `runs/0003_stage2_readout_calibration/train500_pooled_mlp_full16/`

This run is a warmup/calibration baseline. It does not train on hard native labels for the 500 train scenes; LTM30 hard native metrics are used only for validation.

## Outputs

- `best.pt`: best checkpoint by LTM30 primary expected alignment.
- `last.pt`: final checkpoint.
- `summary.json`: run-level summary.
- `best_eval/summary.json`: best-checkpoint LTM30 validation summary.
- `best_eval/ltm30_readout_scores.csv`: per-subset best-checkpoint readout scores.
- `ltm30_readout_scores.csv`: final-checkpoint readout scores.
- `training_history.json`: train-step logs.

## Validation Against LTM30

Lower native geometry errors should have higher readout/register score, so Spearman should be negative. `Expected alignment` below is `-mean_spearman`, where higher is better.

| Method | pose rotation rho | point-map RMSE rho | depth log RMSE rho | Mean rho | Expected alignment |
|---|---:|---:|---:|---:|---:|
| mean-pooled register baseline | -0.5429 | -0.5181 | -0.4990 | -0.5200 | 0.5200 |
| pooled readout best checkpoint | -0.5390 | -0.5581 | -0.4895 | -0.5289 | 0.5289 |
| pooled readout final checkpoint | -0.5219 | -0.5295 | -0.5010 | -0.5175 | 0.5175 |

Best checkpoint:

- checkpoint: `runs/0003_stage2_readout_calibration/train500_pooled_mlp_full16/best.pt`
- epoch: 15
- step: 465
- primary expected alignment: 0.5289

## Interpretation

The train500 pooled readout warmup is stable and produces a real validation signal, but it does not pass the proposed readout gate yet.

Positive signs:

- The best checkpoint slightly improves mean primary expected alignment over mean pooling: `0.5289` vs `0.5200`.
- It improves point-map RMSE correlation clearly: `-0.5581` vs `-0.5181`.
- The training loop, dual-GPU cache path, LTM30 validation, and checkpoint flow are all working.

Limitations:

- The average gain is only `+0.0089`, far below the proposed `+0.10` gate.
- Pose rotation is slightly worse than mean pooling.
- Depth log RMSE is slightly worse than mean pooling.
- Because this run uses online subset-mask warmup rather than hard native labels for train500, it should not be considered the final calibrated readout.

## Decision

Do not promote this pooled readout checkpoint as the locked `0004` selector objective yet.

Recommended next step: generate hard subset native labels for a smaller held-out training subset, then train either pairwise-ranking pooled readout or the 2-layer attention readout. Keep mean-pooled register cosine as the current fallback objective for `0004`.

## Decision Log

- 2026-06-07: Created design after LTM30 showed strong VGGT-native subset-vs-full signal but Stage 1 appearance/sparse-geometry proxies remained insufficient.
- 2026-06-07: Ran `train500_pooled_mlp_full16`; stable but does not pass the readout-improvement gate.
