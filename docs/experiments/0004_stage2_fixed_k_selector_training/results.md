# Results

## Run: `main_v1_meanpool_selector`

- Status: completed.
- Config: `configs/experiments/0004_stage2_fixed_k_selector_training.yaml`
- Manifest: `docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_manifest.json`
- Run dir: `runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/`
- Cache root: `caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512/`
- Cache policy: compact `selector_features.pt` only; no `depth.pt`, no `depth_conf.pt`, no dense pointmap cache.
- Selector objective: full-scene mean-pooled register cosine。
- Budget: `20%` ratio, `K = round(0.20 * N)` with at least one selected frame。
- Cache devices: `cuda:0,cuda:1`
- Train devices: `cuda:0,cuda:1`
- Training epochs / steps: `20` / `2160`
- Training elapsed: `94.94` sec after feature cache finished。
- Feature cache result: `2138/2138` scenes succeeded, `0` failed。

## Dataset

| Dataset | Selected scenes | Notes |
|---|---:|---|
| `bridgedata_v2` | 1000 | `data/processed/bridgedata_v2` |
| `nyuv2` | 549 | `data/processed/nyuv2` |
| `tartanair` | 163 | `data/processed/tartanair` |
| `bonn` | 26 | `data/processed/bonn` |
| `yifei_scannetv2_hf` | 400 | `data/raw/ltm_datasets/yifei_scannetv2_hf` |

| Split | Scenes |
|---|---:|
| train | 1728 |
| val | 205 |
| test | 205 |

Total: `2138` scenes, `105204` sampled frames。

## Training Diagnostics

| Run ID | Epoch | Val Soft Cosine | Val Hard Proxy Cosine | Soft-Hard Gap | Notes |
|---|---:|---:|---:|---:|---|
| `best_hard_proxy.pt` | 18 | 0.992365 | 0.969982 | 0.022383 | best val hard proxy |
| `best_soft.pt` | 20 | 0.993935 | 0.965243 | 0.028691 | same as `last.pt` |
| `last.pt` | 20 | 0.993935 | 0.965243 | 0.028691 | final epoch |

## Proxy Baseline Comparison

对 val split 的 `205` scenes，用相同 cached `register_mean/full_embedding` 计算 20% hard proxy cosine。这个对照不重新跑 VGGT，只用于判断当前 mean-register proxy objective 是否有选择判别力。

| Method | Mean Hard Proxy Cosine | Median | Min | Max |
|---|---:|---:|---:|---:|
| all frames | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| uniform stride | 0.999042 | 0.999463 | 0.993098 | 0.999987 |
| random 20%, 5 seeds | 0.993221 | 0.996456 | 0.904004 | 0.999998 |
| learned topK, `best_hard_proxy.pt` | 0.969982 | 0.978341 | 0.849700 | 0.999973 |
| prefix 20% | 0.942779 | 0.950483 | 0.633729 | 0.999949 |

解读：

- `main_v1` 成功训练出了 selector checkpoint，但没有打过 uniform/random proxy baseline。
- 当前 mean-pooled register proxy 太容易被均匀采样拿到接近满分，因此它不是足够强的 selector promotion 指标。
- `best_hard_proxy.pt` 仍然比 prefix 20% 好，说明模型没有完全失效；但它学习到的排序没有超过简单 coverage baseline。
- 这个结果支持下一步改 objective，而不是直接进入下游 3DGS 验算。

## Preflight

2026-06-08 已完成 `--cache-only --max-scenes 2` smoke cache：

- `cuda:0` 和 `cuda:1` 各缓存 1 个 scene，return code 均为 `0`。
- `selector_features.pt` 写入成功。
- 验证 tensor shape:
  - `frame_features`: `[N, 8193]`
  - `register_mean`: `[N, 2048]`
  - `full_embedding`: `[2048]`
  - dtype: `torch.float16`

## Checkpoints

| Checkpoint | Selection Metric | Status |
|---|---|---|
| `best_soft.pt` | val soft cosine | completed |
| `best_hard_proxy.pt` | val hard proxy cosine | completed |
| `last.pt` | final epoch | completed |

## Decision

Do not promote `main_v1_meanpool_selector` to hard subset VGGT or FastGS/3DGS validation yet. The framework is usable, but this objective/baseline result says mean-pooled register cosine alone is too weak as a selector training target. Recommended next step is a `main_v2` objective: add uniform/random/k-center baseline ranking, or distill from `0003` hard native labels / metric-specific learned readout evidence.

## Run: `main_v2_baseline_rank_selector`

- Status: completed.
- Config: `configs/experiments/0004_stage2_fixed_k_selector_training_main_v2.yaml`
- Cache: reuse `caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512/`
- Run dir: `runs/0004_stage2_fixed_k_selector_training/main_v2_baseline_rank_selector/`
- Objective: baseline-aware rank objective with uniform target CE。
- Budget: `20%` ratio。
- Training epochs / steps: `20` / `2160`
- Training elapsed: `101.15` sec。

1-epoch smoke on full manifest:

| Epoch | Val Soft Cosine | Val Hard Proxy Cosine | Uniform Proxy | Random Proxy | Hard - Uniform | Hard - Random |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.999021 | 0.998906 | 0.999042 | 0.993662 | -0.000137 | 0.005243 |

Interpretation: smoke already recovers near-uniform coverage and beats random on the cache-only proxy. Full 20-epoch run is needed to see whether it can cross uniform rather than merely imitate it.

Full run checkpoints:

| Checkpoint | Epoch | Val Hard Proxy | Uniform Proxy | Random Proxy | Hard - Uniform | Hard - Random | Soft-Hard Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `best_margin.pt` | 16 | 0.998955 | 0.999042 | 0.993662 | -0.000087 | 0.005293 | 0.000072 |
| `best_hard_proxy.pt` | 16 | 0.998955 | 0.999042 | 0.993662 | -0.000087 | 0.005293 | 0.000072 |
| `best_soft.pt` | 3 | 0.998713 | 0.999042 | 0.993662 | -0.000329 | 0.005051 | 0.000326 |
| `last.pt` | 20 | 0.998856 | 0.999042 | 0.993662 | -0.000187 | 0.005193 | 0.000040 |

Proxy eval on `best_margin.pt`, 5 random seeds:

| Method | Mean Hard Proxy Cosine | Median | Min | Max |
|---|---:|---:|---:|---:|
| all frames | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| uniform stride | 0.999042 | 0.999463 | 0.993098 | 0.999987 |
| learned topK, `best_margin.pt` | 0.998955 | 0.999463 | 0.989059 | 0.999987 |
| random 20%, 5 seeds | 0.993221 | 0.996456 | 0.904004 | 0.999998 |
| prefix 20% | 0.942779 | 0.950483 | 0.633729 | 0.999949 |

Interpretation:

- `main_v2` fixes the main_v1 failure mode: learned topK improves from `0.969982` to `0.998955` and beats random by `+0.005293` on val。
- It still does not beat uniform stride: `hard_minus_uniform = -0.000087` at the best checkpoint。
- The soft-hard gap is tiny (`0.000072`), so the remaining issue is not soft/hard mismatch; it is that the mean-register proxy strongly favors uniform coverage。
- Do not promote to hard subset VGGT/3DGS on this proxy alone. The next useful step is a stronger objective or evaluator: hard native labels, register-k-center/ranking candidates, or `0003` learned-readout auxiliary.

## Run: `main_v3_hardnative_candidate_selector`

- Status: completed; val has a small positive signal, test does not pass。
- Config: `configs/experiments/0004_stage2_fixed_k_selector_training_main_v3.yaml`
- Runner: `scripts/run_stage2_selector_hardnative_candidate_training.py`
- Source labels: `runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv`
- Source jobs: `runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json`
- Source token cache: `caches/vggt_omega/0003_stage2_readout_calibration/hardlabel300_full100_80_images512/`
- Candidate set: `uniform20`、`random20_seed000-004`、`contiguous20_seed000`
- Metric: hard-native `target_error`，越低越好。

Precheck on all 300 labeled scenes:

| Split | uniform20 | random20 mean | random20 best-of-5 | contiguous20 | oracle labeled candidate |
|---|---:|---:|---:|---:|---:|
| train | -0.8582 | 0.6240 | -0.5722 | 5.5037 | -1.0555 |
| val | -1.0574 | 0.5989 | -0.6834 | 5.5981 | -1.1961 |
| test | -0.9917 | 0.6833 | -0.5293 | 5.4085 | -1.2233 |

Oracle winner distribution:

| Candidate family | Scenes |
|---|---:|
| `uniform20` | 217 |
| `random20` | 82 |
| `contiguous20` | 1 |

Interpretation: `uniform20` is a strong hard-native baseline, but there is measurable headroom: the labeled oracle is lower than uniform on every split, including test (`-1.2233` vs `-0.9917`)。

Smoke command: 30 scenes, `candidate_set`, 1 epoch, small 2-layer/256 hidden model, single GPU。

Smoke result:

| Split | Learned Error | Uniform Error | Random Mean | Random Best-of-5 | Oracle | Uniform - Learned | Pairwise Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -0.6294 | -0.6422 | 0.5732 | -0.5983 | -0.8239 | -0.0128 | 0.7440 |
| val | 0.1603 | -0.2140 | 0.7135 | 0.1498 | -0.3873 | -0.3744 | 0.8226 |
| test | -1.2483 | -1.2483 | 0.2931 | -0.7226 | -1.2483 | 0.0000 | 0.6667 |

Smoke interpretation:

- Data loading, image-list mask reconstruction, compact feature cache, training loss, checkpoint save/load, and metric reporting are working。
- 1-epoch smoke is not a promotion result. It is intentionally too small to judge whether learned beats uniform。
- The pairwise ranking signal is learnable even in the smoke run: val pairwise accuracy reached `0.8226` after one epoch。

Full run:

- Model: `candidate_set`
- Epochs / steps: `60` / `900`
- Training elapsed: `55.83` sec after feature cache。
- Feature cache: `300` scenes, compact `frame_features` only。
- Best checkpoint: `best_uniform_improvement.pt`
- Best epoch / step: `33` / `495`
- Best val `uniform_minus_learned_error`: `+0.0101`

Best checkpoint metrics:

| Split | Learned Error | Uniform Error | Random Mean | Random Best-of-5 | Oracle | Uniform - Learned | Regret Reduction | Win vs Uniform | Oracle Top1 | Pairwise Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | -1.0555 | -0.8582 | 0.6240 | -0.5722 | -1.0555 | +0.1973 | +0.1973 | 0.2833 | 1.0000 | 0.9682 |
| val | -1.0675 | -1.0574 | 0.5989 | -0.6834 | -1.1961 | +0.0101 | +0.0101 | 0.0333 | 0.8000 | 0.7854 |
| test | -0.7360 | -0.9917 | 0.6833 | -0.5293 | -1.2233 | -0.2557 | -0.2557 | 0.0333 | 0.6333 | 0.7765 |

Full-run interpretation:

- `candidate_set` can overfit the training scenes almost perfectly: train oracle top1 reaches `1.0000`。
- The validation split shows a real but very thin positive window: `uniform_minus_learned_error = +0.0101`。
- The held-out test split fails clearly: `uniform_minus_learned_error = -0.2557`。
- Pairwise accuracy stays non-trivial on val/test, but top-level candidate selection is not calibrated enough to safely deviate from `uniform20`。

Uniform-fallback threshold sweep on the same checkpoint:

| Threshold selection | Split | Learned Error | Uniform Error | Oracle | Uniform - Learned | Deviation Rate |
|---|---|---:|---:|---:|---:|---:|
| best by val | val | -1.0847 | -1.0574 | -1.1961 | +0.0273 | 0.1000 |
| best by val | test | -0.7049 | -0.9917 | -1.2233 | -0.2869 | 0.2333 |
| test oracle scan | test | -0.9917 | -0.9917 | -1.2233 | 0.0000 | 0.0000 |

Threshold interpretation:

- A conservative fallback can improve val from `+0.0101` to `+0.0273` by deviating from uniform on only `10%` of val scenes。
- The same threshold worsens test, and an oracle scan over thresholds says the best test decision is to never deviate from uniform。
- Therefore the current learned deviation signal is not reliable enough for selector promotion。

Frame-score control:

- Run dir: `runs/0004_stage2_fixed_k_selector_training/main_v3_frame_score_candidate_selector/`
- Model: `frame_score`
- Epochs / steps: `60` / `900`
- Training elapsed: `55.50` sec。
- Best checkpoint: `best_uniform_improvement.pt`
- Best val `uniform_minus_learned_error`: `0.0000`

| Split | Learned Error | Uniform Error | Oracle | Uniform - Learned | Win vs Uniform | Oracle Top1 | Pairwise Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -0.8582 | -0.8582 | -1.0555 | 0.0000 | 0.0000 | 0.7167 | 0.6912 |
| val | -1.0574 | -1.0574 | -1.1961 | 0.0000 | 0.0000 | 0.7667 | 0.6820 |
| test | -0.9917 | -0.9917 | -1.2233 | 0.0000 | 0.0000 | 0.7333 | 0.7186 |

Frame-score interpretation:

- `frame_score` 的 best checkpoint 等价于选择 `uniform20`，没有取得正提升。
- 训练过程中它多次偏离 uniform 后变差，因此逐帧平均分数不足以表达 fixed-K subset 的 coverage/diversity 关系。
- 这个结果支持保留 set-aware `candidate_set` 结构，而不是把 selector 简化成每帧独立打分。

Rank-only small candidate-set control:

- Run dir: `runs/0004_stage2_fixed_k_selector_training/main_v3_rankonly_small_candidate_selector_cuda0/`
- Model: `candidate_set`, `2` layers, hidden `256`, dropout `0.2`
- Loss: rank-only, `ce_weight = 0.0`, `min_target_gap = 0.05`
- Train device: `cuda:0`
- Epochs / steps: `60` / `900`
- Training elapsed: `38.27` sec。
- Best val `uniform_minus_learned_error`: `+0.0162`

| Split | Learned Error | Uniform Error | Oracle | Uniform - Learned | Deviation/Win vs Uniform | Oracle Top1 | Pairwise Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -0.9864 | -0.8582 | -1.0555 | +0.1282 | 0.1375 | 0.8000 | 0.9384 |
| val | -1.0737 | -1.0574 | -1.1961 | +0.0162 | 0.0333 | 0.8000 | 0.8199 |
| test | -0.6826 | -0.9917 | -1.2233 | -0.3091 | 0.0000 | 0.6333 | 0.7884 |

Rank-only interpretation:

- 降低容量并去掉 oracle CE 后，val 从 `+0.0101` 提升到 `+0.0162`，但 test 更差。
- 这个 run 说明问题不是 4-layer 过大这么简单；模型仍然学到了 scene-specific deviation，但没有学到稳定的 held-out calibration。
- 双卡 `DataParallel` 对这个小模型曾在 initial eval 触发 CUDA kernel error，单卡 `cuda:0` 复跑成功；该问题记录为 runtime quirk，不影响结果判断。

Post-hoc gate/ensemble scan:

- Script: `scripts/evaluate_stage2_selector_hardnative_gate.py`
- Output: `runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector/ensemble_gate_scan.json`
- Inputs: 4-layer `candidate_set` best checkpoint + 2-layer rank-only checkpoint。
- Rules scanned: single-model uniform fallback and two-model agreement fallback。

| Gate selection | Split | Learned Error | Uniform Error | Oracle | Uniform - Learned | Deviation Rate | Win vs Uniform | Oracle Top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| best by val, all thresholds | val | -1.0847 | -1.0574 | -1.1961 | +0.0273 | 0.1000 | 0.0667 | 0.8000 |
| best by val, all thresholds | test | -0.7049 | -0.9917 | -1.2233 | -0.2869 | 0.2333 | 0.0333 | 0.6000 |
| best by val, positive-margin only | val | -1.0737 | -1.0574 | -1.1961 | +0.0162 | 0.0667 | 0.0333 | 0.8000 |
| best by val, positive-margin only | test | -0.6826 | -0.9917 | -1.2233 | -0.3091 | 0.1667 | 0.0000 | 0.6333 |
| best by test oracle scan | val | -1.0574 | -1.0574 | -1.1961 | 0.0000 | 0.0000 | 0.0000 | 0.7667 |
| best by test oracle scan | test | -0.9917 | -0.9917 | -1.2233 | 0.0000 | 0.0000 | 0.0000 | 0.7333 |

Gate interpretation:

- 按 val 自动选择 gate，最高 val 提升可到 `+0.0273`，但 test 仍明显为负。
- 限制为 positive-margin gate 后，结果退化到 rank-only checkpoint 本身：val `+0.0162`，test `-0.3091`。
- 对 test 做 oracle threshold scan 时，最优规则是完全不偏离 uniform，即 `uniform_minus_learned_error = 0.0000`。
- 因此现有 hardlabel300 / 7-candidate 训练路线没有得到可 promotion 的 selector；当前最可靠 selector 仍是 `uniform20` fallback。

## Current Decision

Do not promote 0004 selector to hard subset VGGT rerun or downstream 3DGS/FastGS validation yet.

The useful result is a negative boundary: `candidate_set` can fit hard-native candidate ranking and finds a small validation signal, but this signal does not survive held-out test. Before another training run, 0004 needs a larger hard-native label set and richer candidates: more scenes, more random seeds, uniform jitter, register/DINO k-center, and an explicit calibrated “safe to deviate from uniform” objective。
