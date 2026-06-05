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
