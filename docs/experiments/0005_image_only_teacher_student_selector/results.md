# 结果

## 当前状态

尚未运行实验。

0005 目前只完成实验定义：从 0004 的 VGGT-feature 后处理 selector，转向 image-only / cheap-feature teacher/student selector。

## 待记录指标

| Run | Student input | Teacher labels | Val uniform - student | Test uniform - student | 结论 |
|---|---|---|---:|---:|---|
| `main_v1_dinov2_candidate_rank` | 待定 | hardlabel300 | 待运行 | 待运行 | 待定 |

## 记录口径

只有当 student 推理时不读取 VGGT-OMEGA tokens/features，结果才计入 0005。

如果某个 run 使用 VGGT-OMEGA per-frame tokens 作为输入，应归入 0004 或另记为 teacher/diagnostic ablation，不能作为 image-only selector 结果。
