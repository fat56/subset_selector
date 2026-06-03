# Metrics Schema

每次 run 的 `metrics.json` 建议包含以下字段。没有计算的字段保留为 `null`。

```json
{
  "embedding": {
    "register_cosine_similarity": null,
    "positive_cosine_loss": null,
    "retrieval_top1_accuracy": null
  },
  "reconstruction": {
    "psnr": null,
    "ssim": null,
    "lpips": null
  },
  "geometry": {
    "pose_ate": null,
    "pose_rpe": null,
    "depth_abs_rel": null
  },
  "efficiency": {
    "subset_size": null,
    "subset_ratio": null,
    "vggt_seconds": null,
    "gs_train_seconds": null
  },
  "correlation": {
    "spearman_rho_register_cosine_vs_psnr": null,
    "pearson_r_register_cosine_vs_psnr": null
  }
}
```

Stage 1 gate 的主指标是 `spearman_rho_register_cosine_vs_psnr`。

