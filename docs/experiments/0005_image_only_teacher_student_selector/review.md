# 复盘

## 当前判断

0005 值得单独开线。

原因是 0004 的核心输入是 full scene 的 VGGT-OMEGA compact features，推理阶段仍需要先跑 VGGT，因此不符合“先选子集，再运行 VGGT/3DGS”的原始目标。

0005 的目标更明确：用 VGGT-OMEGA teacher 生成监督，但训练一个推理时只看原始图像或 cheap image features 的 student selector。

## 建议路线

第一步不要直接做复杂 streaming selector。先跑 Main V1 batch cheap-feature candidate selector，回答一个基础问题：

> cheap image features 是否足以预测 hard-native candidate ranking？

如果这个问题都不成立，streaming memory 和 marginal gain 设计也很难成立。

如果 Main V1 有正信号，再推进：

- Main V2: streaming memory selector。
- Main V3: marginal-gain teacher/student。

## 当前风险

- `uniform20` 是很强 baseline，student 很容易学到看似合理但 test 变差的 deviation。
- hardlabel300 样本太少，仍可能出现 0004 的 val/test 反转。
- cheap image backbone 的语义特征未必等价于三维 coverage。

## 下一步

- 实现 cheap image feature cache。
- 先复用 hardlabel300 做 smoke。
- 若 smoke 有信号，再规划 hardlabel1000 + richer candidates。
