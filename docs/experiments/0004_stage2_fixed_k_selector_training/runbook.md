# Runbook

## 当前状态

`0004/main_v1` 已进入第一版实现和训练阶段。目标是先得到一个固定 `20%` ratio 的 selector checkpoint，训练指标以 `mean-pooled register cosine` proxy 为主；hard subset VGGT 重跑和 FastGS/3DGS 下游验算暂时后置。

关键路径：

- Config: `configs/experiments/0004_stage2_fixed_k_selector_training.yaml`
- Manifest: `docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_manifest.json`
- Manifest summary: `docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_summary.md`
- Cache root: `caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512`
- Run dir: `runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector`

## 前置条件

- VGGT-OMEGA 512 checkpoint 可用。
- `data/processed` 可访问，且包含 `bridgedata_v2`、`nyuv2`、`tartanair`、`bonn`。
- ScanNet 使用 `data/raw/ltm_datasets/yifei_scannetv2_hf/scannetv2/scans/*/color`。
- 双卡训练默认使用 `cuda:0,cuda:1`。
- 本轮不生成 `depth.pt`、`depth_conf.pt` 或 dense VGGT output cache。

## 生成 Manifest

```bash
python scripts/prepare_stage2_selector_main_v1.py --manifest-stem main_v1
```

当前 `main_v1` 统计：

| Dataset | Selected scenes |
|---|---:|
| `bridgedata_v2` | 1000 |
| `nyuv2` | 549 |
| `tartanair` | 163 |
| `bonn` | 26 |
| `yifei_scannetv2_hf` | 400 |

Split:

| Split | Scenes |
|---|---:|
| train | 1728 |
| val | 205 |
| test | 205 |

总计 `2138` scenes、`105204` sampled frames。每个 scene 最多 `64` 帧，最少 `12` 帧。

## 启动训练

推荐用 tmux 启动完整 cache + training：

```bash
mkdir -p runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector

tmux new-session -d -s selector0004_main_v1 '
cd /home/m/project/ltm/selector &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_selector_training.py \
  --manifest docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_manifest.json \
  --cache-root caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512 \
  --run-dir runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector \
  --checkpoint 512 \
  --image-resolution 512 \
  --mode balanced \
  --feature-dtype float16 \
  --cache-devices cuda:0,cuda:1 \
  --train-devices cuda:0,cuda:1 \
  --epochs 20 \
  --batch-size 16 \
  --num-workers 4 \
  2>&1 | tee runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/tmux.log
'
```

如果只需要重训 selector，不重建 feature cache：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_selector_training.py \
  --skip-cache \
  --manifest docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_manifest.json \
  --cache-root caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512 \
  --run-dir runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector \
  --train-devices cuda:0,cuda:1 \
  --epochs 20 \
  --batch-size 16 \
  --num-workers 4
```

## 监控

tmux:

```bash
tmux ls
tmux capture-pane -pt selector0004_main_v1:0 -S -80
```

Feature cache logs:

```bash
tail -n 20 runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/feature_cache_cuda0.log
tail -n 20 runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/feature_cache_cuda1.log
```

Training log:

```bash
tail -n 50 runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/tmux.log
```

主要事件：

- `feature_cache_start`: 两张卡开始分片 cache。
- `feature_cache_done`: 某张卡分片 cache 完成。
- `train_step`: selector training 已进入稳定训练阶段，日志里包含 `steps_per_sec` 和 `eta_sec`。
- `eval`: 每个 epoch 的 val soft/hard proxy cosine。
- `done`: 训练完成并写出 summary。

## 输出

训练成功后应产生：

- `feature_index.json`: scene 到 `selector_features.pt` 的索引。
- `best_soft.pt`: 按 val soft cosine 选择的 checkpoint。
- `best_hard_proxy.pt`: 按 val hard proxy cosine 选择的 checkpoint。
- `last.pt`: 最后一轮 checkpoint。
- `summary.json`: 训练摘要。
- `train_config.json` 和 `config.json`: 训练参数记录。

## Proxy Baseline Eval

训练后用同一份 compact cache 评估 learned selector 与简单 baselines：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/evaluate_stage2_selector_proxy.py \
  --feature-index runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/feature_index.json \
  --checkpoint runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/best_hard_proxy.pt \
  --out runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/proxy_eval.json \
  --split val \
  --ratio 0.20 \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda:0
```

当前 `main_v1` 的 val proxy 对照显示 learned topK 低于 uniform/random，因此不进入 hard subset VGGT 或 FastGS/3DGS 验算。

## Main V2 Baseline-Rank Run

`main_v2` 复用 `main_v1` compact cache，不重新跑 VGGT：

```bash
mkdir -p runs/0004_stage2_fixed_k_selector_training/main_v2_baseline_rank_selector

tmux new-session -d -s selector0004_main_v2 '
cd /home/m/project/ltm/selector &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_selector_training.py \
  --manifest docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_manifest.json \
  --cache-root caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512 \
  --run-dir runs/0004_stage2_fixed_k_selector_training/main_v2_baseline_rank_selector \
  --skip-cache \
  --train-devices cuda:0,cuda:1 \
  --epochs 20 \
  --batch-size 16 \
  --num-workers 4 \
  --objective baseline_rank \
  --pos-weight 0.2 \
  --nce-weight 0.0 \
  --rank-weight 1.0 \
  --uniform-ce-weight 0.2 \
  --rank-margin 0.005 \
  --temperature-start 0.7 \
  --temperature-end 0.2 \
  2>&1 | tee runs/0004_stage2_fixed_k_selector_training/main_v2_baseline_rank_selector/tmux.log
'
```

训练后评估：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/evaluate_stage2_selector_proxy.py \
  --feature-index runs/0004_stage2_fixed_k_selector_training/main_v2_baseline_rank_selector/feature_index.json \
  --checkpoint runs/0004_stage2_fixed_k_selector_training/main_v2_baseline_rank_selector/best_margin.pt \
  --out runs/0004_stage2_fixed_k_selector_training/main_v2_baseline_rank_selector/proxy_eval.json \
  --split val \
  --ratio 0.20 \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda:0
```

`main_v2` 的 primary diagnostic 是 `hard_minus_uniform`。只有 learned topK 至少达到 uniform stride，才值得进入 hard subset VGGT rerun。

## 判断口径

本轮只判断 selector 是否值得进入 hard validation：

- val `hard_proxy_cosine` 是否随训练稳定提升。
- `soft_hard_gap` 是否可控，没有持续扩大。
- top-K proxy 是否明显强于随机/均匀 baseline。当前脚本先产出 learned proxy，baseline 对照可作为下一步补充。

只有当 proxy 结果有明确提升，再进入 hard subset VGGT rerun 和 FastGS/3DGS 验算。
