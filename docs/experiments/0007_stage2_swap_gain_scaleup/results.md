# Results

## Summary

No runs yet. This experiment is planned as the scale-up of `0006` direct swap-gain regression from `300 scenes x 8 swaps` to about `1000 scenes x 8 swaps`.

## Planned Runs

| Run ID | Scenes | Swaps/scene | Feature | Seeds | Status |
|---|---:|---:|---|---|---|
| `swapgain1000_single8_smoke20` | 20 | 8 | DINOv2-S global | n/a | planned |
| `swapgain1000_single8` | 1000 | 8 | DINOv2-S global | n/a | planned |
| `swap_gain_regressor_global_dino_seed20260619-20260623` | 1000 | 8 | DINOv2-S global | 5 | planned |

## Space Budget

Current baseline before 0007 cache:

- Free disk: about `814G`.
- `0006` observed cost: `300 x 8` swap cache used about `103G`.
- `0007` estimate: `1000 x 8` swap cache about `343G`, plus any full/uniform reference cache not reused.

Record actual usage here after smoke and after full label generation.

## Teacher Diagnostics

To fill after label generation:

| Split | Scenes | Swap best win vs uniform | Uniform minus best swap mean | Oracle family notes |
|---|---:|---:|---:|---|
| train | pending | pending | pending | pending |
| val | pending | pending | pending | pending |
| test | pending | pending | pending | pending |

## Student Results

| Seed | Val threshold | Val delta | Test delta | Test win | Test deviation | Gain MAE | Sign acc | Pair acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260619 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 20260620 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 20260621 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 20260622 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 20260623 | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Promotion Summary

To fill:

- Mean test delta:
- Median test delta:
- Worst seed:
- Positive seeds:
- Dataset-wise WildRGBD delta:
- Dataset-wise DL3DV delta:
- Pass/fail:

## Observations

Record implementation gaps, data issues, split sensitivity, and any cache failures here.
