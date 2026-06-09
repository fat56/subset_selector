# Experiment Index

| ID | Stage | Status | Question | Config | Review |
|---|---|---|---|---|---|
| `0001_stage1_register_quality_gate` | stage1 | planned | Does register similarity correlate with 20% FastGS quality? | [config](../../configs/experiments/0001_stage1_register_quality_gate.yaml) | [review](0001_stage1_register_quality_gate/review.md) |
| `0002_ltm30_pose_depth_validation` | validation-data | completed | Can 30 pose/depth scenes provide a stronger geometry validation set? | [manifest](0002_ltm30_pose_depth_validation/manifest.json) | [results](0002_ltm30_pose_depth_validation/results.md) |
| `0003_stage2_readout_calibration` | stage2 | design-draft | Can a trained readout improve the VGGT-native geometry proxy before selector training? | not created yet | [review](0003_stage2_readout_calibration/review.md) |
| `0004_stage2_fixed_k_selector_training` | stage2 | design-draft | Can a fixed-K learned selector beat non-learned baselines after readout/proxy calibration? | not created yet | [review](0004_stage2_fixed_k_selector_training/review.md) |
| `0005_image_only_teacher_student_selector` | stage2 | design-draft | Can a cheap-feature student selector choose fixed-K images before VGGT inference? | not created yet | [review](0005_image_only_teacher_student_selector/review.md) |
