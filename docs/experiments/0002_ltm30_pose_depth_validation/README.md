# LTM30 Pose/Depth 验证集

## 元数据

- 实验 ID: `0002_ltm30_pose_depth_validation`
- 状态: 已准备
- 创建日期: 2026-06-07
- 数据根目录: `data/raw/ltm_datasets -> /home/m/dataset/ltm_datasets`
- Manifest 文件: `docs/experiments/0002_ltm30_pose_depth_validation/manifest.json`
- 准备脚本: `scripts/prepare_ltm30_validation.py`
- 验证脚本: `scripts/run_ltm30_pose_depth_validation.py`
- 结果文档: `docs/experiments/0002_ltm30_pose_depth_validation/results.md`

## 目的

这组 30 个 scene 用来做比 mipnerf360/tandt/db 更强的几何验证：每个入选 frame 都必须有 RGB、depth 和 pose/intrinsics。后续可以把同一 scene 的 `full`、`random20`、`uniform20` 分别送入 VGGT-OMEGA，比较 register token/native geometry 与真实 depth/pose 一致性的关系。

当前 manifest 不复制数据，只保存指向 `data/raw/ltm_datasets` 的相对路径，因此依赖本机数据 symlink。

## 选择策略

- 每个 scene 先找所有带 pose 的合格 frame。
- `full` 从合格 frame 中按时间/编号顺序做 uniform downsample，最多 200 张。
- `random20` 从 `full` 内用 scene-keyed deterministic random 抽 20%。
- `uniform20` 从 `full` 内按顺序均匀抽 20%。
- 模型输入时仍应按 `full_index` 或原始 frame 顺序排序，避免输入顺序引入额外变量。

本次生成结果：

- 30 个 scenes。
- 3000 个 full frames。
- 600 个 random20 frames。
- 600 个 uniform20 frames。
- 每个 scene 为 100 full frames 和 20 subset frames。

## 数据处理

优先级是“真实/直接 depth + pose”高于“COLMAP pseudo depth + pose”，再高于“pose-only”：

- WildRGBD: `rgb/*.jpg` + `depth/*.png` + `metadata/*.npz`，其中 `.npz` 已校验包含 `camera_pose` 和 `camera_intrinsics`。本次 30 个 scene 全部来自这里，因为它直接满足 RGB/depth/pose。
- DL3DV: `images_8` + `transforms.json` + `colmap_depth_480p/depth_maps/*.photometric.bin`，脚本已能识别为候选，但 depth 是 COLMAP photometric pseudo depth，本次未优先选入。
- ScanNet: 当前本地 `yifei_scannetv2_hf` 子集有 `color/pose/intrinsic`，没有 depth 目录。脚本默认不纳入，除非显式加 `--include-pose-only`。
- MegaDepth: 当前本地结构有 `dense*/imgs` 和 `dense*/depths/*.h5`，但未找到稳定的 pose/camera metadata 文件；在补 MegaDepth pose parser 前不纳入。

## 重新生成

```bash
scripts/prepare_ltm30_validation.py
```

主要参数：

```bash
scripts/prepare_ltm30_validation.py \
  --scene-count 30 \
  --max-full-frames 200 \
  --subset-ratio 0.20 \
  --random-seed 20260607
```

输出文件：

- `manifest.json`: 后续实验使用的完整 frame/split manifest。
- `scenes.csv`: scene 级摘要，方便人工检查。
- `summary.json`: 机器可读摘要。
- `summary.md`: 人类可读摘要。

## 后续用途

后续 VGGT-OMEGA/register-token 验证建议按 scene 内比较，而不是跨数据集平均：

- `full` 作为该 scene 的参考输入。
- `random20` 与 `uniform20` 分别作为 20% 子集输入。
- 几何指标优先比较 depth/point-map/pose consistency；PSNR/SSIM 只作为补充渲染质量指标。

## 已完成验证

已完成 30 scene VGGT-OMEGA 验证。为提高 scene 内相关性稳定性，验证脚本在 manifest
的 `random20` 基础上扩展到 5 个 random seeds，并加入 `uniform20`：

```bash
/home/m/project/ltm/vggt-omega/.venv/bin/python \
  scripts/run_ltm30_pose_depth_validation.py \
  --random-seeds 5
```

210/210 cache jobs 成功。核心结果见 `results.md` 和 `native_geometry/`。
