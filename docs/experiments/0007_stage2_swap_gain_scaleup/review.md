# Review

## Decision

Pending.

## Questions

- Did scaling from 300 to 1000 scenes improve validation-to-test calibration?
- Did direct gain regression remain the strongest student formulation?
- Were gains balanced across WildRGBD and DL3DV?
- Was the teacher oracle still strong after scale-up?
- Did disk usage stay within the planned budget?

## Evidence

- Proposal: `docs/experiments/0007_stage2_swap_gain_scaleup/proposal.md`
- Runbook: `docs/experiments/0007_stage2_swap_gain_scaleup/runbook.md`
- Results: `docs/experiments/0007_stage2_swap_gain_scaleup/results.md`
- Config: `configs/experiments/0007_stage2_swap_gain_scaleup.yaml`

## Next Actions

- Fill after the smoke and full scale-up runs.
