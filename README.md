# VGGT-OMEGA Subset Selector

VGGT-OMEGA register token 驱动的数据子集选择研究工作区。

当前阶段的重点不是直接堆训练代码，而是先把项目组织成适合长期迭代的形态：每个实验有固定方案、配置、runbook、结果和复盘；每次运行有本地 manifest、指标文件和索引；关键 stage gate 能被清楚追踪。

研究目标与事实核查见 [initial_plan_docs/VGGT_OMEGA_subset_selection_research_report.md](initial_plan_docs/VGGT_OMEGA_subset_selection_research_report.md)。第一阶段必须先验证：

```text
VGGT-OMEGA register/readout embedding 距离越小
    => sparse-view 3DGS 重建质量越好
```

如果 Stage 1 相关性 gate 不通过，就不进入 selector 训练。

## Project Layout

```text
.
├── configs/                 # base、stage、baseline、experiment YAML
│   ├── baselines/           # random/uniform/k-center 等基线配置
│   ├── experiments/         # 每次实验的可复现配置
│   └── stages/              # Stage 0/1/2... 的共享设置
├── data/                    # 数据布局说明；真实数据默认不进 git
├── docs/
│   ├── adr/                 # 架构决策记录
│   ├── experiments/         # 实验方案、流程、结果、复盘
│   ├── integrations/        # VGGT-OMEGA 等外部依赖集成说明
│   ├── registry/            # experiment/run 索引与指标 schema
│   ├── templates/           # 新实验和 run 记录模板
│   ├── roadmap.md           # stage gate 路线图
│   └── iteration_log.md     # 人类可读的长期迭代日志
├── external/                # 指向 sibling 外部仓库的软链接
├── initial_plan_docs/       # 初始 proposal、review、research report
├── runs/                    # 每次本地运行的轻量 manifest 与指标索引
├── scripts/                 # smoke test 和 CLI 包装脚本
├── src/vggt_omega_selector/ # 研究代码包；当前只放工程脚手架
└── tests/                   # 针对脚手架和后续核心模块的测试
```

这个结构参考了 `fat56/VFM_GS` 的长期实验组织方式：配置固化在 `configs/experiments`，实验叙事在 `docs/experiments`，运行产物放在 `runs` 或外部输出目录，并通过 manifest 与 ledger 串起来。

## Iteration Workflow

1. 新建实验文档和配置：

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage new-experiment \
  --id 0002_my_ablation \
  --title "My ablation" \
  --stage stage1
```

2. 按 `docs/experiments/<id>/runbook.md` 跑 baseline 或分析脚本。

3. 为每次运行固化本地记录：

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage record-run \
  --experiment 0001_stage1_register_quality_gate \
  --stage stage1 \
  --method random_k \
  --dataset sample_scene_set \
  --config configs/experiments/0001_stage1_register_quality_gate.yaml \
  --notes "initial dry run"
```

该命令会创建：

```text
runs/<experiment>/<timestamp_method>/
├── manifest.yaml
├── metrics.json
├── notes.md
└── config.yaml
```

并追加 [docs/registry/run_ledger.csv](docs/registry/run_ledger.csv)。

4. 将核心指标填回 `runs/.../metrics.json` 与 `docs/experiments/<id>/results.md`，再在 `review.md` 写结论和下一步。

## Quick Checks

```bash
python -m compileall src tests
PYTHONPATH=src python -m vggt_omega_selector.cli.manage --help
PYTHONPATH=src python -m vggt_omega_selector.cli.manage vggt-preflight
bash scripts/smoke_test.sh
```

安装 editable 包后也可以直接使用 console script：

```bash
python -m pip install -e .
vggt-selector --help
```

## VGGT-OMEGA Integration

本仓库通过软链接和子进程集成 sibling 仓库 `/home/m/project/ltm/vggt-omega`：

```text
external/vggt-omega -> ../../vggt-omega
```

selector 环境不需要安装 torch。实际 VGGT 推理由 `external/vggt-omega/.venv/bin/python` 执行，输出缓存回本项目：

```bash
PYTHONPATH=src python -m vggt_omega_selector.cli.manage vggt-cache \
  --images path/to/a.png path/to/b.png \
  --output-dir caches/vggt_omega/example_scene/run_0001 \
  --checkpoint 512
```

详见 [docs/integrations/vggt_omega.md](docs/integrations/vggt_omega.md)。

## Git Policy

- `docs/`、`configs/`、`src/`、`scripts/`、轻量 `runs/**/manifest.yaml` 与指标文件适合进入 git。
- 原始数据、VGGT/3DGS 缓存、checkpoint、渲染图、大型日志默认不进 git，只在 manifest 中记录路径和 checksum。
- 每次结论性迭代建议至少提交：实验配置、run manifest、关键指标摘要、review。
