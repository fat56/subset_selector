# Iteration Log

长期记录项目层面的关键变化。单次运行细节写入 `runs/<experiment>/<run_id>/manifest.yaml`，实验结论写入 `docs/experiments/<id>/review.md`。

| Date | Iteration | Change | Evidence | Next |
|---|---|---|---|---|
| 2026-06-03 | 0000 | 建立长期迭代工程框架 | `README.md`, `docs/`, `configs/`, `runs/`, `src/` | 关闭 Stage 1 前置依赖，准备 baseline |
| 2026-06-05 | 0001 | 搭建 Stage 1 20% ratio + FastGS scaffold | `stage1-prepare`, `fastgs-preflight`, `configs/integrations/fastgs.yaml` | 登记真实数据集并跑 FastGS 质量验证 |
| 2026-06-05 | 0002 | 登记 `/home/m/project/ltm/3dgsdata` 并验证 binary sparse subset probe | `data/raw/3dgsdata`, `data/datasets.yaml`, `/tmp/selector_stage1_probe` | 等 GPU 空余后准备全量 Stage 1 runs |
