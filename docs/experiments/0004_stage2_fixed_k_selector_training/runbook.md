# 运行手册

## 当前状态

`0004/main_v1` 已进入第一版实现和训练阶段。目标是先得到一个固定 `20%` ratio 的 selector checkpoint，训练指标以 `mean-pooled register cosine` proxy 为主；hard subset VGGT 重跑和 FastGS/3DGS 下游验算暂时后置。

关键路径：

- 配置: `configs/experiments/0004_stage2_fixed_k_selector_training.yaml`
- Manifest: `docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_manifest.json`
- Manifest 摘要: `docs/experiments/0004_stage2_fixed_k_selector_training/main_v1_summary.md`
- Cache 根目录: `caches/vggt_omega/0004_stage2_fixed_k_selector_training/main_v1_features512`
- 运行目录: `runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector`

## 前置条件

- VGGT-OMEGA 512 checkpoint 可用。
- `data/processed` 可访问，且包含 `bridgedata_v2`、`nyuv2`、`tartanair`、`bonn`。
- ScanNet 使用 `data/raw/ltm_datasets/yifei_scannetv2_hf/scannetv2/scans/*/color`。
- 双卡训练默认使用 `cuda:0,cuda:1`。
- 本轮不生成 `depth.pt`、`depth_conf.pt` 或 dense VGGT output cache。

## 生成 manifest

```bash
python scripts/prepare_stage2_selector_main_v1.py --manifest-stem main_v1
```

当前 `main_v1` 统计：

| 数据集 | 入选 scenes |
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

Feature cache 日志:

```bash
tail -n 20 runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/feature_cache_cuda0.log
tail -n 20 runs/0004_stage2_fixed_k_selector_training/main_v1_meanpool_selector/feature_cache_cuda1.log
```

训练日志:

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

## Proxy baseline 评估

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

## Main V2 baseline-rank 运行

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

当前完整 run 已完成：`best_margin.pt` 的 `hard_minus_uniform = -0.000087`，未过 promotion gate，但已经明显高于 random。

## Main V3 hard-native candidate ranking 运行

`main_v3` 不再用 mean-register proxy cosine 作为训练目标，而是复用 `0003` hardlabel300 的 VGGT-native pseudo-label。它在每个 scene 内比较 `uniform20`、`random20_seed000-004`、`contiguous20_seed000` 这些已重跑过 VGGT 的候选子集，训练 selector 预测哪个候选的 `target_error` 更低。

Smoke:

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_selector_hardnative_candidate_training.py \
  --run-dir runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector_smoke \
  --model-kind candidate_set \
  --limit-scenes 30 \
  --epochs 1 \
  --batch-size 4 \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 4 \
  --train-devices cuda:0 \
  --log-every-steps 1
```

正式 run:

```bash
mkdir -p runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector

tmux new-session -d -s selector0004_main_v3 '
cd /home/m/project/ltm/selector &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_selector_hardnative_candidate_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --run-dir runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector \
  --candidate-tag 20 \
  --model-kind candidate_set \
  --train-devices cuda:0,cuda:1 \
  --epochs 60 \
  --batch-size 16 \
  --hidden-dim 512 \
  --num-layers 4 \
  --num-heads 8 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --rank-weight 1.0 \
  --ce-weight 0.3 \
  --min-target-gap 0.02 \
  --target-gap-scale 1.0 \
  --log-every-steps 20 \
  2>&1 | tee runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector/tmux.log
'
```

监控：

```bash
tmux capture-pane -pt selector0004_main_v3:0 -S -120
tail -n 60 runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector/tmux.log
```

主要指标：

- `target_error` 越低越好。
- `uniform_minus_learned_error > 0` 表示 learned selector 的平均 hard-native target error 低于 `uniform20`。
- `learned_regret < uniform_regret` 表示 learned 更接近当前 labeled candidate oracle。
- `mean_pool_cosine_select_error` 是旧 proxy 的候选选择对照；预检查中它弱于 uniform，不作为 promotion 依据。

Frame-score 对照：

```bash
mkdir -p runs/0004_stage2_fixed_k_selector_training/main_v3_frame_score_candidate_selector

tmux new-session -d -s selector0004_main_v3_framescore '
cd /home/m/project/ltm/selector &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_selector_hardnative_candidate_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --run-dir runs/0004_stage2_fixed_k_selector_training/main_v3_frame_score_candidate_selector \
  --candidate-tag 20 \
  --model-kind frame_score \
  --train-devices cuda:0,cuda:1 \
  --epochs 60 \
  --batch-size 16 \
  --hidden-dim 512 \
  --num-layers 4 \
  --num-heads 8 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --rank-weight 1.0 \
  --ce-weight 0.3 \
  --min-target-gap 0.02 \
  --target-gap-scale 1.0 \
  --log-every-steps 20 \
  2>&1 | tee runs/0004_stage2_fixed_k_selector_training/main_v3_frame_score_candidate_selector/tmux.log
'
```

小型 rank-only `candidate_set` 对照：

```bash
mkdir -p runs/0004_stage2_fixed_k_selector_training/main_v3_rankonly_small_candidate_selector_cuda0

tmux new-session -d -s selector0004_main_v3_ranksmall0 '
cd /home/m/project/ltm/selector &&
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/run_stage2_selector_hardnative_candidate_training.py \
  --labels-csv runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/hardlabel_train_labels.csv \
  --cache-jobs-json runs/0003_stage2_readout_calibration/hardlabel300_labels_full100_80/cache_jobs.json \
  --run-dir runs/0004_stage2_fixed_k_selector_training/main_v3_rankonly_small_candidate_selector_cuda0 \
  --candidate-tag 20 \
  --model-kind candidate_set \
  --train-devices cuda:0 \
  --epochs 60 \
  --batch-size 16 \
  --hidden-dim 256 \
  --num-layers 2 \
  --num-heads 4 \
  --dropout 0.2 \
  --lr 0.00005 \
  --weight-decay 0.001 \
  --rank-weight 1.0 \
  --ce-weight 0.0 \
  --min-target-gap 0.05 \
  --target-gap-scale 1.0 \
  --log-every-steps 20 \
  2>&1 | tee runs/0004_stage2_fixed_k_selector_training/main_v3_rankonly_small_candidate_selector_cuda0/tmux.log
'
```

Gate/ensemble 扫描：

```bash
PYTHONPATH=src /home/m/project/ltm/vggt-omega/.venv/bin/python scripts/evaluate_stage2_selector_hardnative_gate.py \
  --device cuda:0 \
  --out runs/0004_stage2_fixed_k_selector_training/main_v3_hardnative_candidate_selector/ensemble_gate_scan.json
```

这个扫描只用于 post-hoc calibration diagnostic。promotion 只能看按 val 选出的 gate 在 test 上是否仍为正；不能用 test oracle scan 作为真实 selector 结果。

## 判断口径

本轮只判断 selector 是否值得进入 hard validation：

- val `hard_proxy_cosine` 是否随训练稳定提升。
- `soft_hard_gap` 是否可控，没有持续扩大。
- top-K proxy 是否明显强于随机/均匀 baseline。当前脚本先产出 learned proxy，baseline 对照可作为下一步补充。

只有当 proxy 结果有明确提升，再进入 hard subset VGGT rerun 和 FastGS/3DGS 验算。
