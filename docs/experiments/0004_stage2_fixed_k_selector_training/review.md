# Review

## Decision

Pending design review.

## Evidence Needed

- Stage 2.0 readout/proxy decision from `0003_stage2_readout_calibration`。
- Readout head choice: frozen checkpoint path or explicit mean-pooling fallback。
- Final selector input feature list。
- Final fixed `K` or ratio for first training run。
- Hard validation results versus Stage 1 baselines。
- FastGS/3DGS validation metrics。

## Current Recommendation

Adopt the soft-token proxy plus hard-subset validation plan for the first implementation. Start with `L_pos` and optionally `L_nce`; keep coverage, redundancy, quality, depth, and pose auxiliary losses disabled until diagnostics show a specific failure mode. Do not start with RL. Do not jointly train an unconstrained readout and selector until the frozen-readout baseline is measured.

## Next Actions

- Wait for `0003_stage2_readout_calibration` decision。
- Import frozen readout checkpoint or mean-pooling fallback from `0003`。
- If accepted, create the Stage 2 config and minimal cache/train/eval implementation。
