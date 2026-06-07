# Review

## Review Status

Not reviewed.

## Review Questions

- Does the readout target use VGGT-native geometry consistency rather than appearance-only metrics?
- Is LTM30 held out from training, or is any split change explicitly documented?
- Does the trained readout improve over mean-pooled register cosine on scene-held-out validation?
- Are direct sensor pose metrics excluded from the pass/fail gate?
- Is the selected readout checkpoint frozen before `0004_stage2_fixed_k_selector_training` starts?

## Risks

- The readout may overfit if trained on only a small number of scenes.
- Ranking labels based on VGGT-native metrics may reproduce VGGT's internal bias rather than external metric geometry.
- Soft embedding distance can still disagree with hard subset VGGT reruns.
- A high-capacity readout can hide selector failures if trained jointly too early.

## Current Recommendation

Proceed with design and dataset-building. Do not start fixed-K selector training until either a readout passes the gate or the selector experiment explicitly chooses mean pooling as its baseline objective.
