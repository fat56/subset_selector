# Results

## Run: `main_v1_meanpool_selector`

- Status: framework ready; full cache/training run starting.
- Config: `configs/experiments/0004_stage2_fixed_k_selector_training.yaml`
- Manifest: `docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_manifest.json`
- Run dir: `runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/`
- Cache root: `caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512/`
- Cache policy: compact `selector_features.pt` only; no `depth.pt`, no `depth_conf.pt`, no dense pointmap cache.
- Selector objective: full-scene mean-pooled register cosine。
- Budget: `20%` ratio, `K = round(0.20 * N)` with at least one selected frame。

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
| `main_v1_meanpool_selector` | pending | pending | pending | pending | 等待 tmux run 产出首轮 eval |

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
| `best_soft.pt` | val soft cosine | pending |
| `best_hard_proxy.pt` | val hard proxy cosine | pending |
| `last.pt` | final epoch | pending |

## Decision

Pending. 本轮先看 selector proxy 是否稳定提升；hard subset VGGT rerun 和 FastGS/3DGS validation 暂不作为本次训练阻塞项。
