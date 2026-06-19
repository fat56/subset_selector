# Results

## Summary

`0007` completed the planned scale-up from `0006` direct swap-gain regression to `1000 scenes x 8 single swaps`. The teacher oracle remained strong, but the image-only student did not become more stable. Validation-selected thresholds produced mean held-out test delta `-0.1703`, median `-0.0203`, and only `2/5` positive seeds, so this run fails promotion.

The main failure mode is threshold/generalization instability rather than missing teacher headroom. The best single swap beats `uniform20` on `91.2%` of scenes, but the learned gain model's test sign and pairwise accuracies stay near chance, and two seeds over-swap badly after selecting permissive validation thresholds.

## Runs

| Run ID | Scenes | Swaps/scene | Feature | Seeds | Status |
|---|---:|---:|---|---|---|
| `swapgain1000_single8_smoke20` | 20 | 8 | DINOv2-S global | n/a | complete |
| `swapgain1000_single8` | 1000 | 8 | DINOv2-S global | n/a | complete |
| `swap_gain_regressor_global_dino_seed20260619-20260623` | 1000 | 8 | DINOv2-S global | 5 | complete, not promoted |

## Space Usage

Baseline before the 0007 cache run was about `814G` free on `/`. Final state after labels and training:

| Artifact | Actual size |
|---|---:|
| Full VGGT cache: `swapgain1000_single8_images512` | `596G` |
| Smoke VGGT cache: `swapgain1000_single8_smoke20_images512` | `11G` |
| DINO image-only feature cache | `96M` |
| Run outputs, logs, labels, checkpoints | `236M` |
| Final free disk on `/` | `208G` |

The original `343G` estimate was too low because the full run included full/reference work in addition to the single-swap subsets. Training itself added little space pressure; the large disk cost is the completed VGGT tensor cache.

## Teacher Diagnostics

The full label run produced `9000` label rows from `1000` scenes, balanced as `500` DL3DV and `500` WildRGBD. The run completed after fixing truncated-image tolerance in the VGGT cache runners.

| Metric | Value |
|---|---:|
| Scenes with at least one swap candidate | `1000` |
| Label rows | `9000` |
| Swap oracle rate vs `uniform20` | `0.912` |
| Uniform minus best swap mean | `2.2821` |
| Uniform minus best swap min | `-2.1469` |
| Uniform minus best swap max | `8.5501` |
| Pair target gain mean | `-0.5667` |
| Pair target gain positive fraction | `0.4209` |

Oracle family counts:

| Oracle family | Scenes |
|---|---:|
| `swapgain20` | `912` |
| `uniform20` | `88` |

## Student Results

Primary decision uses the validation-selected threshold from each seed and reports the corresponding held-out test result.

| Seed | Val threshold | Val delta | Test delta | Test win | Test deviation | Gain MAE | Sign acc | Pair acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260619 | `2.0` | `+0.0968` | `+0.0476` | `0.02` | `0.03` | `2.1512` | `0.5131` | `0.5289` | Conservative, slightly positive. |
| 20260620 | `2.0` | `+0.0372` | `-0.0203` | `0.02` | `0.05` | `2.0353` | `0.5707` | `0.5364` | Near neutral, slightly negative. |
| 20260621 | `0.8` | `+0.1908` | `-0.5225` | `0.18` | `0.58` | `2.0925` | `0.5316` | `0.5127` | Over-swaps badly on test. |
| 20260622 | `1.5` | `+0.0736` | `+0.0379` | `0.18` | `0.40` | `2.1345` | `0.5386` | `0.5115` | Positive, but high deviation. |
| 20260623 | `1.0` | `+0.2064` | `-0.3941` | `0.15` | `0.37` | `2.0072` | `0.5296` | `0.5217` | Over-swaps on test. |

Aggregate:

| Metric | Value |
|---|---:|
| Mean test delta | `-0.1703` |
| Median test delta | `-0.0203` |
| Worst seed | `-0.5225` |
| Best seed | `+0.0476` |
| Positive seeds | `2/5` |
| Mean test win rate | `0.1100` |
| Mean test deviation rate | `0.2860` |
| Mean gain MAE | `2.0841` |
| Mean sign accuracy | `0.5367` |
| Mean pairwise accuracy | `0.5223` |

Dataset-wise test delta under each seed's validation-selected threshold:

| Dataset | Mean delta | Median delta | Positive seeds | Worst | Best |
|---|---:|---:|---:|---:|---:|
| `DL3DV-ALL-480P` | `-0.2014` | `-0.0100` | `2/5` | `-0.8448` | `+0.1283` |
| `wildrgbd_harrison` | `-0.1391` | `-0.0306` | `2/5` | `-0.7336` | `+0.0645` |

Test-oracle threshold scans were diagnostic only and did not change the promotion decision. They found `+0.2249` for seed `20260619`, `+0.0491` for seed `20260622`, and `0.0` for the other three seeds by falling back to `uniform20`.

## Promotion Summary

- Mean test delta: `-0.1703`, below the required `+0.05`.
- Median test delta: `-0.0203`, below the required `> 0`.
- Positive seeds: `2/5`, below the required `4/5`.
- Worst seed: `-0.5225`, below the required `>= -0.02`.
- Dataset-wise results are negative on both DL3DV and WildRGBD.
- Decision: fail promotion; keep labels, but do not scale more of the same direct global-DINO setup.

## Observations

- The teacher labels are useful: swap oracle headroom is larger than in the student results.
- Scaling the dataset did not solve the core image-only student problem. The model fits training well, but validation-selected thresholding is not reliable on held-out scenes.
- The split sensitivity in `0006` was not just small-data noise. At 1000 scenes, the gain regressor still has weak test ranking/sign calibration.
- Disk usage remained stable during training, but the full VGGT cache is now the main space risk at `596G`. Since labels are written, this cache is optional for future recomputation rather than required for student training.
