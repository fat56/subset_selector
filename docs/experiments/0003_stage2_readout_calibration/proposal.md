# Stage 2.0 Readout Calibration

## Metadata

- Experiment ID: `0003_stage2_readout_calibration`
- Stage: `stage2`
- Status: hard-label pilot completed; readout not promoted yet
- Created: 2026-06-07
- Config: implemented with `hardlabel100_full100_80`
- Depends on: `0001_stage1_register_quality_gate`, `0002_ltm30_pose_depth_validation`
- Feeds into: `0004_stage2_fixed_k_selector_training`

## Question

在 Stage 1 的 mean-pooled register-token proxy 只得到中等信号之后，是否应该先训练或校准一个 lightweight readout head，把 VGGT-OMEGA 的 camera/register tokens 转成更可靠的场景级 3D understanding proxy，再把它冻结给 Stage 2 selector 使用？

## Current Evidence

现有证据支持开始 Stage 2，但支持的是一个收窄后的 Stage 2.0：先做 readout calibration 和 VGGT-native geometry proxy，而不是直接进入最终 selector/FastGS/VLA 结论。

### What Stage 1 Did Not Prove

`0001_stage1_register_quality_gate` 中，mean-pooled register token 和 PSNR/SSIM/LPIPS 的 scene-wise 相关性偏弱；FastGS appearance quality 对 register token 的 3D understanding 能力不是最佳目标。

补跑的 sparse geometry proxy 也没有支持 mean-pooled register token 直接作为 selector training signal：对 `colmap_sparse_full_scene` 的 accuracy/completeness/Chamfer/F-score，mean Spearman 接近 0，best-method 一致率很低。

### What LTM30 Did Prove

`0002_ltm30_pose_depth_validation` 在 30 个 pose/depth scene 上提供了更强的验证。mean-pooled register cosine 和 VGGT-native subset-vs-full consistency 有明确方向：

| Metric | Mean Spearman | Sign | Best Match |
|---|---:|---:|---:|
| `pose_rotation_mean_deg` | -0.5429 | 29/30 | 21/30 |
| `pointmap_rmse_norm` | -0.5181 | 28/30 | 22/30 |
| `depth_log_rmse` | -0.4990 | 28/30 | 25/30 |
| `depth_absrel_mean` | -0.4819 | 30/30 | 24/30 |
| `pose_center_rmse_norm` | -0.4857 | 26/30 | 19/30 |

因此，register tokens 确实含有可用于判断 subset 3D consistency 的信息；不足之处在于当前读法太粗糙。

### Hard-Label Pilot Result

`hardlabel100_full100_80` 已完成：100 个训练 scene，1,300 个 VGGT cache jobs，1,200 个 subset-vs-full hard native labels。pairwise-ranking pooled MLP readout 的 best checkpoint 在 LTM30 held-out validation 上达到：

| Method | Expected alignment |
|---|---:|
| mean-pooled register baseline | 0.5200 |
| `train500_full16` warmup best | 0.5289 |
| `hardlabel100_full100_80` best | 0.5594 |

该结果说明 hard native labels 确实比 weak online mask warmup 更有效，但提升为 `+0.0394`，仍未达到原定 `+0.10` readout gate。因此 pooled MLP readout 暂不晋级为 `0004` 的锁定 selector objective；mean-pooled register cosine 仍保留为 fallback，下一步更合理的是复用 hard labels 训练 attention readout 或扩大 hard-label 数据。

### External GT Caveat

WildRGBD sensor depth 有弱但可用的方向性；direct sensor pose GT 目前不可靠，不应作为主训练目标。第一版 readout 应优先对齐 VGGT-native subset-vs-full depth、derived point-map、pose rotation consistency，并把 sensor depth 作为 sanity check。

## Hypothesis

一个冻结 VGGT-OMEGA 之上的小型 RegisterReadoutHead，可以比 mean pooling 更稳定地把 camera/register tokens 读成场景级 embedding 或质量分数。该 readout 的输出距离或 score 应该和下列 VGGT-native errors 在 scene 内负相关：

- `pose_rotation_mean_deg`
- `pointmap_rmse_norm`
- `depth_log_rmse`
- `depth_absrel_mean`
- `pose_center_rmse_norm`

如果 readout calibration 成功，则 `0004_stage2_fixed_k_selector_training` 可以使用 frozen readout 作为 selector 的稳定 proxy loss。

## Scope

本实验只训练或校准 readout head，不训练 selector。

包括：

- 使用 frozen VGGT-OMEGA cache，不更新 VGGT-OMEGA。
- 训练 parameter-free mean pooling、small MLP readout、attention readout 三类读法的对照。
- 用 scene-held-out split 验证 readout distance/score 与 VGGT-native geometry metrics 的相关性。
- 评估 readout 是否比 mean pooling 更适合作为 selector loss。

不包括：

- 不训练 fixed-K selector；该内容顺延到 `0004_stage2_fixed_k_selector_training`。
- 不用 VLA/downstream task 作本阶段主指标。
- 不把 direct external pose GT 作为主目标，除非先完成 pose convention audit。
- 不端到端更新 VGGT-OMEGA。

## Readout Candidates

### Baseline A: Mean-Pooled Register Cosine

当前 Stage 1/LTM30 使用的 parameter-free baseline：

```text
register_tokens: [B, N, R, C]
z = mean(register_tokens, dim=[B, N, R])
z = L2Normalize(z)
similarity = cosine(z_subset, z_full)
```

优点是无训练、稳定、容易复现；缺点是丢失 frame order、camera token、register slot structure、coverage/redundancy 信息。

### Baseline B: Pooled MLP Readout

```text
per-scene summary = concat(
    mean(camera_tokens),
    mean(register_tokens),
    max(register_tokens),
    std(register_tokens),
    optional scalar summaries
)
summary -> LayerNorm -> MLP -> z or quality score
```

预计参数量约 `1-3M`。适合作为第一个训练式 readout，能判断“少量可学习校准是否已经足够”。

### Candidate C: Attention RegisterReadoutHead

VGGT-OMEGA cache 中 observed token shape 为：

```text
camera/register tokens: [B, N, R + 1, C]
C = 2048
R = 16 in current checkpoint cache, but implementation must read from manifest/checkpoint.
```

推荐结构：

```text
tokens
    -> reshape [B, N * (R + 1), C]
    -> Linear(C, 512)
    -> concat 1 learnable readout token
    -> 2 x Pre-LN TransformerEncoderBlock
    -> take readout token
    -> Linear(512, 256 or 512)
    -> L2 normalize
```

参数量估算：

| Variant | Main dims | Approx params |
|---|---|---:|
| pooled MLP | pooled 2048-d summaries -> 256/512 | 1-3M |
| attention readout, 2 blocks | 2048 -> 512, 2 Transformer blocks | 7-8M |
| attention readout, 4 blocks | 2048 -> 512, 4 Transformer blocks | 13-14M |

这些参数量相对 VGGT-OMEGA 很小；训练成本主要来自 subset label/cache 生成，而不是 head 本身。

## Training Targets

第一版推荐把 readout 训练成 ranking/calibration head，而不是只做 scene identity contrastive learning。

### Pairwise Ranking Target

同一 scene 内，如果 subset A 的 native error 明显低于 subset B，则 readout 应给 A 更高质量或更接近 full 的 embedding：

```text
score(S) = -distance(readout(tokens_S), readout(tokens_full))
L_rank = max(0, margin - score(S_good) + score(S_bad))
```

`S_good` / `S_bad` 可由组合指标排序得到：

```text
target_error =
    w_rot * zscore(pose_rotation_mean_deg)
  + w_point * zscore(pointmap_rmse_norm)
  + w_depth * zscore(depth_log_rmse)
```

MVP 权重：

```text
w_rot = 1.0
w_point = 1.0
w_depth = 1.0
```

### Metric Regression Target

辅助训练一个 scalar quality/error head：

```text
readout(tokens_subset, tokens_full) -> predicted_error
L_reg = SmoothL1(predicted_error, target_error)
```

这能让 readout 的分数更可解释，但不能单独替代 hard subset validation。

### Embedding Alignment Target

训练 subset embedding 接近 full embedding：

```text
L_pos = 1 - cosine(z_subset, stopgrad(z_full))
```

仅使用 `L_pos` 可能会过度鼓励所有 subset 都靠近 full，无法区分好坏 subset。推荐只作为辅助项，与 ranking/regression 搭配。

## Data Requirement

当前 LTM30 的 30 scenes x 6 subset = 180 subset pairs，更适合作为 held-out validation，不足以训练一个泛化 readout。

推荐数据规模：

| Purpose | Scenes | Subsets per scene | Pairs |
|---|---:|---:|---:|
| smoke test | 50-100 | 10-20 | 500-2,000 |
| usable small readout | 100-200 | 20-50 | 2,000-10,000 |
| robust readout | 200-500 | 20-50 | 4,000-25,000 |

Split 必须按 scene 划分。不能让同一个 scene 的不同 subset 同时进入 train 和 validation。

子集生成建议：

- ratios: `10%`, `20%`, `30%`
- methods: random seeds, uniform stride, contiguous windows, farthest/k-center candidates, high-overlap low-diversity negatives
- full: 每个 scene 最多 200 张图，排除测试集或使用当前 LTM full split 口径

LTM30 当前 30 个 scene 建议保留为 validation set；如果需要训练，优先从 `/home/m/dataset/ltm_datasets` 中抽取额外 pose/depth scenes。

### Local Data Audit

2026-06-07 对 `data/raw/ltm_datasets -> /home/m/dataset/ltm_datasets` 做了一次只读扫描。当前已有数据足够构建较可靠的 readout calibration set，甚至不需要动用全部数据。

现有 scanner 可直接识别的候选如下：

| Dataset | Usable scenes | Frames | Depth | Pose | Notes |
|---|---:|---:|---|---|---|
| `wildrgbd_harrison` | 1,998 | 199,800 | sensor depth png | npz `camera_pose` | 每个 scene 100 帧；最适合作 readout 训练和 sensor-depth sanity。 |
| `DL3DV-ALL-480P` | 976 | 78,080 | COLMAP photometric depth | NeRF `transforms.json` c2w | 每个 scene 80 帧；适合补充室外/大场景与 SfM-style pseudo depth。 |
| `yifei_scannetv2_hf` | 1,510 | 479,891 | none in local copy | txt camera pose | pose-only；可做 VGGT-native pseudo-label 和 selector robustness，不适合作 sensor-depth sanity。 |
| `MegaDepth_v1` | not counted yet | n/a | likely available after format support | SfM-style | 本地占用约 203G，但当前 scanner 未接入；先不作为 MVP 依赖。 |

合计当前 scanner 可见 `4,484` 个 pose scene，`757,771` 帧；其中 `2,974` 个 scene 同时有 depth 和 pose。这个规模已经超过 robust readout 的建议规模 `200-500` scenes / `4k-25k` subset pairs。

因此第一版建议只抽一个可控子集：

- readout train: `300-500` scenes from WildRGBD + DL3DV。
- readout validation: keep current LTM30 held out, optionally add `50-100` extra held-out scenes。
- optional pseudo-label expansion: add ScanNet pose-only scenes only after depth+pose readout baseline works。
- MegaDepth: wait until scanner/format audit is complete。

### Split Policy With Stage 2 Selector

Readout calibration data should be separated from selector evaluation data, and preferably separated from selector training data as well.

Strict policy:

- `readout_train`: used only to train/calibrate the readout.
- `readout_val/test`: used only to choose readout checkpoint and threshold; current LTM30 should live here.
- `selector_train`: used to train the fixed-K selector in `0004`; no scene overlap with `readout_val/test`.
- `selector_val/test`: used for hard subset VGGT/FastGS/VLA validation; no scene overlap with readout or selector training.

If data is scarce, `readout_train` and `selector_train` can overlap because the readout is frozen before selector training, but that weakens the scientific claim. Given the current `ltm_datasets` scale and the user's extra ~500G data budget, the recommended plan is to avoid overlap by default.

The extra ~500G dataset is valuable as a distribution buffer:

- use part of it for `selector_train` so the selector does not only learn readout-calibration scenes;
- reserve a clean final test slice for post-Stage2 hard VGGT/FastGS/VLA validation;
- do not spend all extra data on readout, because readout only needs hundreds of scenes while selector/generalization validation benefits from independent data.

### Selected Train500 MVP

For the first running experiment, use [train500_manifest.json](train500_manifest.json):

| Setting | Value |
|---|---:|
| WildRGBD scenes | 250 |
| DL3DV scenes | 250 |
| Total train scenes | 500 |
| Full-view frames per scene | 16 |
| Total cached training images | 8,000 |
| LTM30 validation scenes excluded | 30 |

Selection policy:

- deterministic random seed: `20260607`
- scene sampling: balanced WildRGBD/DL3DV, excluding current LTM30 validation scenes
- frame sampling: uniform downsample over each scene's eligible frames
- cache scope: full-view camera/register tokens only
- training subsets: online masks sampled from full cached tokens at ratios `0.25`, `0.5`, `0.75`
- validation: LTM30 hard subset VGGT-native metrics remain validation-only

This is a warmup/calibration run designed to enter training quickly. It does not yet generate hard native labels for all 500 training scenes; that heavier follow-up can be enabled after the full-cache readout baseline is measured.

### Hard-Label Full-View Pilot

The `train500_full16` warmup proved the pipeline but did not pass the readout gate. The next experiment should spend compute on a better-aligned target rather than more weakly supervised scenes.

Use `hardlabel100_full100_80`:

| Setting | Value |
|---|---:|
| WildRGBD scenes | 50 |
| DL3DV scenes | 50 |
| Total train scenes | 100 |
| WildRGBD full-view frames | 100 |
| DL3DV full-view frames | 80 |
| Hard subsets per scene | 12 |
| VGGT cache jobs | 1,300 |

Subset methods per scene:

- `random20_seed000` ... `random20_seed004`
- `random50_seed000` ... `random50_seed002`
- `uniform20`
- `uniform50`
- `contiguous20_seed000`
- `contiguous50_seed000`

This produces actual hard native labels:

```text
full images -> VGGT-OMEGA -> full depth/pose/register cache
hard subset images -> VGGT-OMEGA -> subset depth/pose/register cache
same subset images in full cache vs subset cache -> native geometry errors
```

Primary hard-label target:

```text
target_error =
    zscore_by_scene(pose_rotation_mean_deg)
  + zscore_by_scene(pointmap_rmse_norm)
  + zscore_by_scene(depth_log_rmse)
```

The readout should rank lower `target_error` subsets above higher `target_error` subsets. LTM30 remains held-out validation and is not used for training.

## End-to-End Decision

readout 可以融入 Stage 2 selector 的训练图中，但第一版不建议 joint training。

推荐顺序：

1. parameter-free mean pooling baseline。
2. 单独训练/校准 readout。
3. 冻结 readout，训练 `0004` fixed-K selector。
4. 只有当 frozen-readout selector 的 hard validation 稳定后，才尝试 selector + readout joint fine-tuning。

不建议一开始端到端联合训练的原因：

- readout 和 selector 会 co-adapt，selector 可能学会利用 readout 漏洞，而不是选出真实更好的图像。
- readout 目标移动会让 selector loss 不稳定。
- soft relaxed top-K 与 hard VGGT subset 之间本来就有 gap，joint training 会更难定位失败原因。
- 完整穿过 VGGT-OMEGA 和 hard top-K 的端到端训练不现实；可行的“端到端”只是 cached-token selector/readout 端到端。

如果后续做 joint fine-tuning，应使用小学习率、readout distillation anchor、scene-held-out hard validation，并保留 frozen-readout checkpoint 作为回退。

## Metrics

Primary validation metrics:

- scene-wise mean Spearman between readout score and `pose_rotation_mean_deg`
- scene-wise mean Spearman between readout score and `pointmap_rmse_norm`
- scene-wise mean Spearman between readout score and `depth_log_rmse`
- best-score vs best-native-quality match rate
- sign consistency across scenes

Secondary metrics:

- Pearson correlation
- sensor `gt_depth_absrel_mean` correlation
- hard subset rank accuracy among random/uniform/k-center candidates
- calibration error for scalar predicted native error

Do not use direct sensor `gt_pose_*` as a gate until pose convention is audited.

## Gate

This experiment passes only if a trained/calibrated readout improves over mean-pooled register cosine on scene-held-out validation.

Suggested initial pass thresholds:

- Native geometry target mean Spearman improves by at least `+0.10` over mean pooling on two of three primary targets.
- Expected sign is correct on at least `28/30` LTM30 validation scenes for the strongest target, or equivalent held-out set rate.
- Best-method match improves over mean pooling on at least two primary targets.
- No regression larger than `0.05` mean Spearman on sensor depth sanity metric.

If these thresholds are not met, `0004` should start with mean pooling as a baseline objective and treat trained readout as not ready.

## Recommended Next Step

Create a readout-calibration dataset builder before selector training:

1. Keep the existing LTM30 as held-out validation.
2. Sample additional pose/depth scenes from `data/raw/ltm_datasets`.
3. Generate 10-50 candidate subsets per scene.
4. Cache frozen VGGT-OMEGA camera/register/depth/pose outputs.
5. Compute native subset-vs-full metrics.
6. Train pooled MLP and attention readout.
7. Promote the best readout checkpoint into `0004` only after the gate passes.
