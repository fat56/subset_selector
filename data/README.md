# Data Directory

真实数据默认不进入 git。本目录只保留数据布局、registry 和 schema。

推荐场景组织：

```text
data/
├── datasets.yaml
├── raw/
│   └── <dataset>/<scene>/
│       ├── images/
│       ├── sparse/0/           # COLMAP text or binary model for FastGS subsets
│       ├── poses/              # optional
│       ├── depths/             # optional
│       ├── masks/              # optional
│       └── metadata.json
├── interim/                    # resized images, extracted frame lists
├── processed/                  # split manifests, feature manifests
└── external/                   # third-party repos/checkpoints, ignored
```

每个 sample 是完整 scene/video，split 必须按 scene 切分，避免同一场景帧泄漏到不同 split。

Stage 1 当前以 FastGS 作为重建后端。`stage1-prepare` 会把选中的 20% 图片
symlink 到 prepared source，并把 `sparse/0` 的 COLMAP text 或 binary model 过滤
成 FastGS 可读的 text sparse model。

## Local LTM Datasets

大规模训练/验证数据仍然不进 git。当前 LTM 数据集建议以本地 symlink 挂到：

```bash
ln -s /home/m/dataset/ltm_datasets data/raw/ltm_datasets
```

`data/raw/` 已被 `.gitignore` 忽略；仓库只提交脚本、manifest 和文档。当前
`0002_ltm30_pose_depth_validation` 使用这个链接生成 30 个带 RGB/depth/pose
的验证 scene manifest。
