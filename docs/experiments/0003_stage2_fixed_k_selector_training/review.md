# Review

## Decision

Pending design review.

## Evidence Needed

- Stage 1 gate acceptance with concrete date and metrics。
- Readout head choice: frozen checkpoint path or calibration plan。
- Final selector input feature list。
- Final fixed `K` or ratio for first training run。
- Hard validation results versus Stage 1 baselines。
- FastGS/3DGS validation metrics。

## Current Recommendation

Adopt the soft-token proxy plus hard-subset validation plan for the first implementation. Start with `L_pos` and optionally `L_nce`; keep coverage, redundancy, quality, depth, and pose auxiliary losses disabled until diagnostics show a specific failure mode. Do not start with RL. Do not jointly train an unconstrained readout and selector until the frozen-readout baseline is measured.

## Next Actions

- Wait for Stage 1 GPU validation and review decision。
- Decide whether Stage 1 produced a locked readout or Stage 2 needs readout calibration。
- If accepted, create the Stage 2 config and minimal cache/train/eval implementation。
