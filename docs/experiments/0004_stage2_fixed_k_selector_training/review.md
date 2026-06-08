# Review

## Decision

Do not promote `main_v1_meanpool_selector` beyond proxy validation.

`0003_stage2_readout_calibration` 已经证明 learned readout 的 embedding diagnostic 可以明显提升，但 single unified checkpoint 仍未稳定通过严格 gate：hardlabel300 2-layer best embedding 为 `0.6063`，4-layer 降到 `0.5822`，而 conservative mean-pooled register baseline 为 `0.5200`。因此 `0004/main_v1` 不把 learned readout 作为依赖，先用 mean-pooled register cosine 训练 selector。

`main_v1` 的工程目标已经完成：manifest、cache-light feature、4-layer selector、bounded soft top-K 和双卡训练都跑通。但 val proxy baseline 显示 learned top-K 没有打过 uniform/random，因此不能把这个 checkpoint 直接推进到 hard subset VGGT 或 FastGS/3DGS。

## 当前判断

- 先训练 selector 网络本身的方向是对的；本轮已证明框架可跑。
- 本轮采用 cache-light 策略，避免再次写入 `depth.pt`、`depth_conf.pt` 这类大文件。
- ScanNet 使用 `data/raw/ltm_datasets/yifei_scannetv2_hf`，符合当前数据源偏好。
- 下游 3DGS/FastGS 验算继续放一边；当前 proxy 对照已经足够说明 `main_v1` 不值得直接下游验算。
- 下一步应补更有判别力的 objective，例如 baseline ranking、hard native labels、或把 `0003` 的 learned readout evidence 作为 auxiliary signal。

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

`main_v1` 结果：

- Feature cache: `2138/2138` scenes 成功，`0` failed。
- Training: `20` epochs, `2160` steps, training elapsed `94.94` sec。
- `best_hard_proxy.pt`: epoch 18, val hard proxy cosine `0.969982`。
- `best_soft.pt` / `last.pt`: epoch 20, val soft cosine `0.993935`, hard proxy cosine `0.965243`。

Proxy baseline on val:

| Method | Mean Hard Proxy Cosine |
|---|---:|
| uniform stride | 0.999042 |
| random 20%, 5 seeds | 0.993221 |
| learned topK | 0.969982 |
| prefix 20% | 0.942779 |

这个对照说明 mean-pooled register proxy 对均匀覆盖过于友好，无法证明 learned selector 比朴素策略更好。

## Next Actions

- 保留 `main_v1` 作为 framework/proxy negative result。
- 跑 `main_v2_baseline_rank_selector`：复用 `main_v1` compact cache，用 uniform/random baseline-aware objective 训练。
- 对 `main_v2` 先跑同样 cache-only proxy baseline，只有超过 uniform/random 后再进入 hard subset VGGT rerun。

## Main V2 Result

`main_v2` 没有引入新的 VGGT cache 或 3DGS 验算。它只回答一个更小的问题：learned selector 能不能在 mean-register proxy 上至少追平 uniform/random。

训练目标：

- `0.2 * L_pos`
- `1.0 * L_rank_to_best(uniform, random)`
- `0.2 * L_uniform_target_ce`
- `rank_margin = 0.005`

Promotion gate:

- learned topK mean hard proxy cosine `>=` uniform stride。
- learned topK mean hard proxy cosine `>` random 20%。
- soft-hard gap `<= 0.005`。

结果：

- `best_margin.pt`: epoch 16。
- learned topK hard proxy cosine: `0.998955`。
- uniform stride: `0.999042`。
- random 20%, 5 seeds: `0.993221`。
- `hard_minus_uniform = -0.000087`。
- `hard_minus_random = +0.005293`。
- soft-hard gap: `0.000072`。

结论：

- `main_v2` 明显修复了 `main_v1`：learned 从 `0.969982` 提升到 `0.998955`。
- 它高于 random，但仍没有超过 uniform，因此没有通过 promotion gate。
- soft-hard gap 已经很小，说明问题不是 relaxation，而是 mean-register proxy 本身对 uniform coverage 过于友好。
- 下一步不应继续在同一个 proxy 上堆训练轮次；应该换更强 supervision，例如 hard native labels、候选 subset ranking、register-k-center 对照，或 `0003` learned-readout auxiliary。

## Main V3 Direction

`main_v3` 已切换到 `0003` hard-native candidate ranking：

- 复用 hardlabel300 的 `target_error`，不再把 mean-register proxy 当主目标。
- 每个 scene 内比较 `uniform20`、`random20_seed000-004`、`contiguous20_seed000`。
- Primary metric 改为 `uniform_minus_learned_error`，大于 `0` 才表示 learned 的 hard-native target error 低于 uniform。
- 第一版模型为 `candidate_set` scorer，并保留 `frame_score` 作为表达力对照。

Smoke 已通过 30 scenes / 1 epoch：

- 数据加载、candidate mask 重建、compact feature cache、loss、checkpoint save/load 都正常。
- 小样本 val pairwise accuracy 达到 `0.8226`，说明 hard-native ranking signal 能被模型读到。
- smoke 不作为 promotion 结果；下一步需要跑 300 scenes 完整双卡 run，观察 val/test 是否同时超过 uniform。
