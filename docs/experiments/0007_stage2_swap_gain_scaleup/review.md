# Review

## Decision

Fail promotion. `0007` should be kept as a useful teacher-label asset, but the direct global-DINO swap-gain regressor should not be promoted as the next selector policy.

The scale-up answered the main question clearly: more scenes alone did not make the `0006` signal stable. The teacher still has strong oracle headroom, but the student cannot reliably choose when to swap on held-out scenes.

## Answers

| Question | Answer |
|---|---|
| Did scaling from 300 to 1000 scenes improve validation-to-test calibration? | No. Validation-selected thresholds overfit in several seeds; mean test delta was `-0.1703`. |
| Did direct gain regression remain the strongest student formulation? | It remains the best tested formulation so far, but this run shows it is not robust enough with only global DINO image features. |
| Were gains balanced across WildRGBD and DL3DV? | No robust gain on either. DL3DV mean test delta was `-0.2014`; WildRGBD mean test delta was `-0.1391`. |
| Was the teacher oracle still strong after scale-up? | Yes. The best swap beat `uniform20` in `91.2%` of scenes with mean improvement `+2.2821`. |
| Did disk usage stay within the planned budget? | Training was fine, but VGGT cache usage exceeded the estimate: full cache `596G`, final free space `208G`. |

## Evidence

- Proposal: `docs/experiments/0007_stage2_swap_gain_scaleup/proposal.md`
- Runbook: `docs/experiments/0007_stage2_swap_gain_scaleup/runbook.md`
- Results: `docs/experiments/0007_stage2_swap_gain_scaleup/results.md`
- Config: `configs/experiments/0007_stage2_swap_gain_scaleup.yaml`
- Full labels: `runs/0007_stage2_swap_gain_scaleup/swapgain1000_single8/augmented_hardlabel_train_labels.csv`
- Student summaries: `runs/0007_stage2_swap_gain_scaleup/swap_gain_regressor_global_dino_seed*/summary.json`

## Interpretation

The teacher side is not the bottleneck. The 8 single-swap candidates are often meaningfully better than `uniform20`, but many individual swaps are harmful, and the global image-only student does not learn a reliable scene-level accept/reject rule. It can fit training rankings, yet held-out sign and pairwise accuracies stay close to chance.

The two worst seeds selected lower validation thresholds (`0.8` and `1.0`), increased test deviation to `0.58` and `0.37`, and lost heavily. More conservative thresholds avoided catastrophic losses but mostly fell back to `uniform20`, leaving little positive gain.

## Next Actions

- Do not launch another same-form 1000-scene global-DINO direct-gain run before changing the student signal.
- Try richer image-only features next: patch-summary/temporal aggregation, adjacent-frame motion cues, or a small per-frame sequence model with explicit uniform-subset context.
- Add a conservative calibration objective or threshold policy that penalizes false-positive swaps more strongly than missed swaps.
- Reuse the `0007` labels for ablations; avoid regenerating VGGT labels unless a new candidate family is being tested.
- If disk pressure returns, the `596G` full VGGT cache can be deleted after preserving labels, jobs, summaries, and DINO feature cache.
