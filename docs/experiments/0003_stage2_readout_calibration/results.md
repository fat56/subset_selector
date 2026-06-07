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

## Run: `hardlabel100_pooled_mlp_full100_80`

- Date: 2026-06-07
- Train scenes: 100
- Train source: 50 WildRGBD + 50 DL3DV
- Full-view frames per scene: WildRGBD 100, DL3DV 80
- Hard subsets per scene: 12
- VGGT cache jobs: 1,300/1,300 succeeded
- Hard native label rows: 1,200
- Cache devices: `cuda:0,cuda:1`
- Train devices: `cuda:0,cuda:1`
- Model: pooled MLP readout
- Objective: pairwise ranking over same-scene hard subsets, with positive full/subset cosine and symmetric InfoNCE auxiliary losses.
- Validation: LTM30 hard subset validation, 30 scenes x 6 subsets = 180 rows.
- Run dir: `runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80/`

This run uses actual hard native labels rather than online masks. Each subset is rerun through VGGT-OMEGA and compared against the same images inside the full-view cache for `pose_rotation_mean_deg`, `pointmap_rmse_norm`, and `depth_log_rmse`.

## Hard-Label Outputs

- `hardlabel_native_metrics.csv`: subset-vs-full VGGT-native metrics for 1,200 hard subsets.
- `hardlabel_train_labels.csv`: per-scene z-scored training targets and token paths.
- `hardlabel_summary.json`: label counts and target range.
- `best.pt`: best checkpoint by LTM30 primary expected alignment.
- `best_eval/summary.json`: best-checkpoint LTM30 validation summary.
- `best_eval/ltm30_readout_scores.csv`: per-subset best-checkpoint readout scores.
- `last.pt`: final checkpoint.
- `summary.json`: run-level summary.
- `training_history.json`: train-step logs.

## Hard-Label Validation Against LTM30

Lower native geometry errors should have higher readout/register score, so Spearman should be negative. `Expected alignment` below is `-mean_spearman`, where higher is better.

| Method | pose rotation rho | point-map RMSE rho | depth log RMSE rho | Mean rho | Expected alignment |
|---|---:|---:|---:|---:|---:|
| mean-pooled register baseline | -0.5429 | -0.5181 | -0.4990 | -0.5200 | 0.5200 |
| `train500_full16` pooled best | -0.5390 | -0.5581 | -0.4895 | -0.5289 | 0.5289 |
| `hardlabel100` pooled best | -0.5048 | -0.5962 | -0.5771 | -0.5594 | 0.5594 |
| `hardlabel100` pooled final | -0.4724 | -0.5619 | -0.5333 | -0.5225 | 0.5225 |

Best checkpoint:

- checkpoint: `runs/0003_stage2_readout_calibration/hardlabel100_pooled_mlp_full100_80/best.pt`
- epoch: 35
- step: 14,000
- primary expected alignment: 0.5594
- training time after labels: 2,802.3 seconds

## Hard-Label Interpretation

The hard-label pilot is a useful improvement over the warmup, but it is not yet strong enough to promote the pooled MLP readout as the locked Stage 2 selector objective.

Positive signs:

- Mean primary expected alignment improves from the mean-pooled register baseline `0.5200` to `0.5594`.
- The gain over the warmup best checkpoint is larger: `+0.0305`.
- Point-map RMSE and depth log RMSE correlations improve clearly over mean pooling.
- The full hard-label pipeline worked end to end: dual-GPU VGGT cache, native label generation, dual-device readout training, best-checkpoint validation.

Limitations:

- The average gain over mean pooling is `+0.0394`, below the proposed `+0.10` gate.
- Pose rotation correlation regresses from `-0.5429` to `-0.5048`; the model appears to learn point/depth consistency better than pose rotation.
- The final checkpoint is worse than the best checkpoint, so checkpoint selection matters and over-training remains possible.
- The pooled MLP architecture may be too lossy for camera/register token structure even when hard labels are good.

## Hard-Label Decision

Do not promote the pooled MLP hard-label checkpoint as the final `0004` selector objective yet.

Recommended next step: keep this hard-label dataset and train a structured attention readout, or expand hard-label scenes/subset diversity before deciding whether to freeze a readout. Mean-pooled register cosine remains the fallback selector proxy.

## Run: `hardlabel100_attention_multimetric`

- Status: completed.
- Train labels: reuse `hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv`.
- VGGT cache: reused; no new VGGT cache generation.
- Model: cross-attention readout with scene query and three metric queries.
- Metric heads: `pose_rotation_mean_deg`, `pointmap_rmse_norm`, `depth_log_rmse`.
- Objective: per-metric pairwise ranking, full-view score anchor, low-weight embedding alignment and InfoNCE.
- Validation: LTM30 hard subset validation, metric-head expected alignment as primary score.
- Train devices: `cuda:0,cuda:1`
- Epochs: 30
- Steps: 13,500
- Training time: 2,022.5 seconds
- Run dir: `runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric/`

This run tests whether the pooled readout failed mainly because it discarded camera/register token structure.

Validation summary:

| Method | pose rotation rho | point-map RMSE rho | depth log RMSE rho | Mean rho | Expected alignment |
|---|---:|---:|---:|---:|---:|
| mean-pooled register baseline | -0.5429 | -0.5181 | -0.4990 | -0.5200 | 0.5200 |
| `hardlabel100` pooled best | -0.5048 | -0.5962 | -0.5771 | -0.5594 | 0.5594 |
| attention metric-head best | -0.5295 | -0.5505 | -0.6171 | -0.5657 | 0.5657 |
| attention metric-head final | -0.5029 | -0.5162 | -0.5752 | -0.5314 | 0.5314 |
| attention embedding-cosine best diagnostic | -0.5105 | -0.5219 | -0.5733 | -0.5352 | 0.5352 |

Best checkpoint:

- checkpoint: `runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric/best.pt`
- epoch: 25
- step: 11,250
- primary metric-head expected alignment: 0.5657
- embedding-cosine diagnostic expected alignment: 0.5352

Interpretation:

- Attention/multi-metric heads improve over pooled hard-label readout only slightly: `0.5657` vs `0.5594`.
- The strongest gain is depth log RMSE: `-0.6171`, better than pooled `-0.5771`.
- Pose rotation recovers part of the pooled regression but still remains below mean-pooling baseline: `-0.5295` vs `-0.5429`.
- The final checkpoint again drops from the best checkpoint, reinforcing that early stopping is required.
- The result does not pass the strict `+0.10` promotion gate; architecture alone is not the main bottleneck.

Decision:

Do not promote the attention readout as the locked `0004` selector objective. Keep mean-pooled register cosine as the selector fallback. The next readout improvement should prioritize more/diverse hard labels or target audit rather than another small head tweak.

## Run: `hardlabel100_attention_multimetric_ratio20_margin`

- Status: completed.
- Train labels: reuse `hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv`.
- VGGT cache: reused; no new VGGT cache generation.
- Model: same cross-attention multi-metric readout as `hardlabel100_attention_multimetric`.
- Training filter: only methods containing `20`.
- Pair filter: same-scene metric margin at least `0.25` of that scene/metric range.
- Label rows after filter: 700 rows from 100 scenes.
- Training pairs after filter: 3,255 metric pairs.
- Train devices: `cuda:0,cuda:1`
- Epochs: 30
- Steps: 6,090
- Training time: 682.7 seconds
- Run dir: `runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric_ratio20_margin/`

This run tests whether the previous attention result was hurt by mixing 20/50/80/100% subsets while LTM30 validation is 20%-subset only, and by noisy near-tie pseudo-label pairs.

Validation summary:

| Method | pose rotation rho | point-map RMSE rho | depth log RMSE rho | Mean rho | Expected alignment |
|---|---:|---:|---:|---:|---:|
| mean-pooled register baseline | -0.5429 | -0.5181 | -0.4990 | -0.5200 | 0.5200 |
| attention metric-head best | -0.5295 | -0.5505 | -0.6171 | -0.5657 | 0.5657 |
| ratio20/margin metric-head best | -0.3676 | -0.3581 | -0.4324 | -0.3860 | 0.3860 |
| ratio20/margin metric-head final | -0.3295 | -0.2895 | -0.3543 | -0.3244 | 0.3244 |
| ratio20/margin embedding diagnostic peak | -0.5410 | -0.5886 | -0.5981 | -0.5759 | 0.5759 |
| ratio20/margin embedding diagnostic final | -0.5619 | -0.5695 | -0.5600 | -0.5638 | 0.5638 |

Best metric-head checkpoint:

- checkpoint: `runs/0003_stage2_readout_calibration/hardlabel100_attention_multimetric_ratio20_margin/best.pt`
- epoch: 7
- primary metric-head expected alignment: 0.3860
- embedding-cosine diagnostic at this checkpoint: 0.5422

Embedding diagnostic note:

- The best embedding diagnostic occurred at epoch 16 with expected alignment `0.5759`.
- The training loop selected checkpoints by metric-head alignment, so the epoch-16 embedding-best model was not retained as `best.pt`.
- This diagnostic is still useful as evidence about objective/checkpoint selection, but it is not a promotable checkpoint artifact.

Interpretation:

- The ratio-20/margin filtering is a clear negative result for the metric heads: `0.3860` is far below both the mean-pooled baseline `0.5200` and prior attention best `0.5657`.
- The embedding diagnostic remains competitive and slightly exceeds prior attention head best at its peak, but that signal is not what the current primary objective selects.
- The result suggests the bottleneck is not simply subset-ratio mismatch or near-tie pair noise. The multi-metric head objective itself may be too brittle on the narrowed 20%-only label set.
- Do not start a larger 20%-only multi-metric run. The next useful ablation is either single-target training or changing checkpoint selection/objective to preserve embedding-best models.

Decision:

Do not promote this checkpoint. Treat the run as evidence against 20%-only multi-metric head training with a `0.25` margin filter. If continuing locally, prefer a short single-target ablation starting with `depth_log_rmse` or a rerun that checkpoints by embedding expected alignment.

## Run: `hardlabel100_attention_depth_ratio20_margin`

- Status: completed.
- Train labels: reuse `hardlabel100_pooled_mlp_full100_80/hardlabel_train_labels.csv`.
- VGGT cache: reused; no new VGGT cache generation.
- Model: cross-attention readout with one metric head.
- Target metric: `depth_log_rmse`.
- Training filter: only methods containing `20`.
- Pair filter: same-scene metric margin at least `0.25` of the depth metric range.
- Label rows after filter: 700 rows from 100 scenes.
- Training pairs after filter: 1,131 depth pairs.
- Train devices: `cuda:0,cuda:1`
- Epochs: 30
- Steps: 2,100
- Training time: 252.8 seconds
- Run dir: `runs/0003_stage2_readout_calibration/hardlabel100_attention_depth_ratio20_margin/`

This run tests whether the ratio20/margin regression was caused by multi-target head interference.

Validation summary:

| Method | depth log RMSE rho | Expected alignment |
|---|---:|---:|
| mean-pooled register baseline | -0.4990 | 0.4990 |
| attention metric-head best, all metrics | -0.6171 | 0.6171 |
| ratio20/margin metric-head best, all metrics | -0.4324 | 0.4324 |
| ratio20/margin depth-only head best | -0.4248 | 0.4248 |
| ratio20/margin depth-only head final | -0.3010 | 0.3010 |
| ratio20/margin depth-only embedding diagnostic peak | -0.5543 | 0.5543 |
| ratio20/margin depth-only embedding diagnostic final | -0.5048 | 0.5048 |

Best checkpoint:

- checkpoint: `runs/0003_stage2_readout_calibration/hardlabel100_attention_depth_ratio20_margin/best.pt`
- epoch: 20
- depth-head expected alignment: 0.4248
- embedding diagnostic at this checkpoint: 0.5124

Interpretation:

- Depth-only training did not rescue the ratio20/margin metric-head setup; its best depth correlation is slightly worse than the multi-metric ratio20/margin depth head.
- The prior all-ratio attention model remains much better on depth (`0.6171` expected alignment), so removing higher-ratio subsets likely removed useful supervision rather than just cleaning noise.
- The embedding diagnostic remains stronger than the metric head, but still below the all-ratio attention depth head.
- Do not spend more time on pose-only/point-only under this exact 20%-only/margin recipe. The next useful change should alter the objective/checkpointing or return to all-ratio labels with single-target heads.

Decision:

Do not promote. Stop the current 20%-only/margin branch after this depth-only check.

## Decision Log

- 2026-06-07: Created design after LTM30 showed strong VGGT-native subset-vs-full signal but Stage 1 appearance/sparse-geometry proxies remained insufficient.
- 2026-06-07: Ran `train500_pooled_mlp_full16`; stable but does not pass the readout-improvement gate.
- 2026-06-07: Ran `hardlabel100_pooled_mlp_full100_80`; hard native labels improve the pooled readout materially over the warmup, but still do not pass the strict `+0.10` gate.
- 2026-06-07: Ran `hardlabel100_attention_multimetric`; attention and separate metric heads slightly improve best alignment to `0.5657`, still below the strict promotion gate.
- 2026-06-07: Ran `hardlabel100_attention_multimetric_ratio20_margin`; metric-head alignment regressed to `0.3860`, while embedding diagnostics peaked at `0.5759` without a retained embedding-best checkpoint.
- 2026-06-07: Ran `hardlabel100_attention_depth_ratio20_margin`; depth-only head still underperformed (`0.4248`), so the 20%-only/margin branch is not worth expanding as-is.
