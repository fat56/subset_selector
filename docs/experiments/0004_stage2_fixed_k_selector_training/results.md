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
