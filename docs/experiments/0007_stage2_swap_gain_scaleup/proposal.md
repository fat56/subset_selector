# 1000-Scene Swap-Gain Scale-Up

## Metadata

- Experiment ID: `0007_stage2_swap_gain_scaleup`
- Stage: `stage2`
- Status: planned
- Created: 2026-06-19
- Config: `configs/experiments/0007_stage2_swap_gain_scaleup.yaml`

## Question

Does the weak but repeatable signal from `0006` direct swap-gain regression become materially more stable when the teacher label set is scaled from `300 scenes x 8 single swaps` to about `1000 scenes x 8 single swaps`?

## Hypothesis

`0006` showed the first val-selected image-only student with positive held-out mean improvement over `uniform20`: test delta mean `+0.0535`, `4/5` seeds positive, but median only `+0.0082`. The likely bottleneck is not teacher headroom; dense single-swap labels beat `uniform20` in `90.7%` of scenes. The bottleneck is student generalization and threshold calibration from too few scenes.

Scaling to about 1000 scenes should reduce split sensitivity and make the direct gain regressor's validation-selected threshold more reliable. If this is true, positive test delta should hold across seeds without relying on test-oracle threshold scans.

## Method

The experiment keeps the successful `0006` formulation:

```text
base subset = uniform20(scene)
candidate = uniform20 - removed_frame + added_frame
gain = target_error(uniform20) - target_error(candidate)
student predicts gain from image-only features
```

The first scale target is:

- `1000` scenes.
- `8` single-swap candidates per scene.
- `20%` fixed-K budget.
- VGGT-OMEGA used only for offline teacher labels.
- Student input remains image-only DINO feature plus image statistics.

Important implementation note: current `0006` label generation reuses `hardlabel300` / `richer300` sources. A true 1000-scene run first needs a 1000-scene source manifest and image-only feature cache. Do not assume `--limit-scenes 1000` on the current script is enough.

## Data Plan

Preferred 1000-scene source mix:

| Dataset | Scenes | Reason |
|---|---:|---|
| WildRGBD Harrison | 500 | Object-centric scenes with sensor depth; strong continuity with `0003/0006`. |
| DL3DV | 500 | More diverse real-world videos; reduces overfitting to WildRGBD object scans. |

The LTM30 validation scenes must remain excluded. Scene-level split is `80/10/10`, stratified by dataset.

If either source pool is short after filtering, keep total near 1000 by drawing from the larger source and record the final mix in `results.md`.

## Label Generation

For each selected scene:

1. Sample the full frame set with the same policy as hard-label experiments:
   - WildRGBD: up to `100` frames.
   - DL3DV: up to `80` frames.
2. Build `full` and `uniform20` image lists.
3. Use DINOv2 image-only features to rank candidate additions outside `uniform20`.
4. Generate `8` single-swap candidates: `swapgain20_dino1_rank000-007`.
5. Run VGGT-OMEGA cache for full/uniform/swap candidates as needed.
6. Compute VGGT-native depth/pose/point-map metrics.
7. Write `augmented_hardlabel_train_labels.csv` and `augmented_cache_jobs.json`.

If full/uniform metrics already exist for a scene, reuse them. If not, include them in this experiment's cache budget.

## Student Training

Primary model:

- Script: `scripts/run_stage2_image_only_swap_gain_regressor.py`
- Feature cache: DINOv2-S/ViT-S global image-only features.
- Seeds: `20260619`, `20260620`, `20260621`, `20260622`, `20260623`.
- Epochs: `120`.
- Objective: gain regression + sign loss + pairwise rank loss.
- Threshold selection: validation-selected threshold only; test-oracle scans are diagnostic only.

Secondary model only if primary passes or is close:

- Patch-summary temporal DINO features.
- Same direct gain objective.

## Metrics

Primary metric:

- `test_uniform_minus_learned_error_mean` across five seeds.

Promotion criteria:

- Mean test delta `>= +0.05`.
- Median test delta `> 0`.
- At least `4/5` seeds positive.
- Worst seed `>= -0.02`.
- Val-selected rule only; no test-oracle threshold in the pass/fail decision.

Secondary metrics:

- Test win rate vs `uniform20`.
- Test deviation rate.
- Gain MAE.
- Gain sign accuracy.
- Pairwise gain accuracy.
- Test-oracle threshold delta, diagnostic only.
- Dataset-wise deltas for WildRGBD and DL3DV.

## Resource Budget

Current disk state before planning: about `814G` free on `/`.

Observed cache cost from `0006`:

- `300 scenes x 8 single swaps` used about `103G`.
- Linear estimate for `1000 scenes x 8 single swaps`: about `343G`.

Planned budget:

| Component | Estimate |
|---|---:|
| New single-swap VGGT cache | `340G` |
| Full/uniform reference cache if not reused | extra, to be measured by smoke |
| Image-only DINO features | `< 2G` |
| Run outputs/checkpoints/logs | `< 5G` |
| Safety reserve after cache | target `>= 250G` free |

The experiment is space-feasible if the first smoke confirms per-scene cache cost near the `0006` estimate. If free disk drops below `250G`, stop before launching another VGGT cache batch and clean completed cache or reduce scale.

## Decision Rule

Pass:

- Direct gain regressor meets the primary promotion criteria above.
- Dataset-wise results do not reveal one source carrying all gains while the other is strongly negative.

Continue but do not promote:

- Mean test delta is positive but median is near zero or seed sensitivity remains high.
- In that case, keep the labels and try stronger image-only features or a calibrated conservative threshold objective.

Fail:

- Best validation-selected strategy falls back to `uniform20`, or mean test delta is non-positive.
- If teacher oracle remains strong, the failure is student-side; next step should change features/model, not generate more similar labels.

## Risks

- Existing label script is hardlabel300-centric; a manifest-based 1000-scene label builder may be needed before the VGGT run.
- Full/uniform reference cache may add more disk than the `340G` single-swap estimate.
- More data may reduce variance but not solve feature insufficiency.
- Validation threshold may still overfit if val split is only 100 scenes; keep per-dataset validation summaries.
