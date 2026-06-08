# Review

## Decision

Proceed with `main_v1_meanpool_selector`.

`0003_stage2_readout_calibration` 已经证明 learned readout 的 embedding diagnostic 可以明显提升，但 single unified checkpoint 仍未稳定通过严格 gate：hardlabel300 2-layer best embedding 为 `0.6063`，4-layer 降到 `0.5822`，而 conservative mean-pooled register baseline 为 `0.5200`。因此 `0004/main_v1` 不把 learned readout 作为依赖，先用 mean-pooled register cosine 训练 selector。

## 当前判断

- 先训练 selector 网络本身，比继续加 readout 容量更值得。
- 本轮采用 cache-light 策略，避免再次写入 `depth.pt`、`depth_conf.pt` 这类大文件。
- ScanNet 使用 `data/raw/ltm_datasets/yifei_scannetv2_hf`，符合当前数据源偏好。
- 下游 3DGS/FastGS 验算先放一边，避免把第一版 selector 训练变成过重流程。
- 如果 `main_v1` 的 proxy 指标没有提升，再考虑补 baseline 对照、ranking refinement 或重新引入 hard labels。

## Evidence

`main_v1` 已生成：

- `2138` scenes。
- `105204` sampled frames。
- train/val/test = `1728/205/205`。
- 数据来源包含 `bridgedata_v2`、`nyuv2`、`tartanair`、`bonn` 和 `yifei_scannetv2_hf`。

实现已覆盖：

- Manifest builder。
- Compact per-scene VGGT feature cache。
- 4-layer Transformer fixed-ratio selector。
- Bounded sigmoid soft top-K mask。
- Soft cosine + symmetric InfoNCE training objective。
- Val soft cosine 与 hard proxy cosine checkpointing。

## Next Actions

- 启动 tmux 双卡完整 cache + training。
- 等训练进入稳定 `train_step` 后，根据日志估算结束时间。
- 训练结束后追加 `best_soft.pt`、`best_hard_proxy.pt` 和 `summary.json` 到 results。
- 如果 proxy 结果有明显提升，再补 hard subset VGGT rerun 和 baseline 对照。
