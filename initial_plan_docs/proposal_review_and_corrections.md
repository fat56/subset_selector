# Proposal 评价、更正与建议（精简版）

> 日期：2026-06-03
> 对象：[`VGGT_OMEGA_subset_selection_proposal.md`](VGGT_OMEGA_subset_selection_proposal.md)
> 详版：[`VGGT_OMEGA_subset_selection_research_report.md`](VGGT_OMEGA_subset_selection_research_report.md)
> 依据：一次多源联网事实核查（抓取一手页面 + 对抗式投票）。⚠️ 本轮 `WebSearch` 工具全程故障，结论以直取的 arXiv/官方 repo/主页等**一手来源**为准，缺独立第三方对抗检索。

---

## 一、总体评价：**方向扎实、自我批判到位、引用无造假**

- ✅ **基石成立**：方案核心模型 **VGGT-Ω 真实存在**（arXiv:[2605.15195](https://arxiv.org/abs/2605.15195)，CVPR 2026 Oral，原 VGGT 团队，官方 repo/主页齐全），其 **registers + register attention** 经论文摘要 + 主页 + 代码三重核实，proposal 第 11、39–52 行的结构描述**基本准确**。
- ✅ **引用诚实**：被点名的所有支撑性先例（registers 起源 ICLR'24、DUSt3R、Sener-Savarese/CRAIG/GradMatch、Skeletal Sets）**全部真实**，没有引用造假或张冠李戴。
- ✅ **难得的清醒**：proposal **自己**在第 35、488 行就点出了"register 距离是否真等于重建质量需要验证"这一最关键软肋，并明智地拒绝从 RL 起步、坚持先建强 baseline —— 这是这份方案最值得肯定的地方。

> 一句话：**这是一份可以立项的方案，但有一个必须先迈过的门槛和五处需要更正/补强的地方。**

---

## 二、事实层面的更正（逐条，带证据）

| # | proposal 原述 | 更正 | 证据 / 置信度 |
|---|---|---|---|
| C1 | （隐含）VGGT-OMEGA 是确定可引的工作 | ✅ **属实**，无需改 —— 但注意它是 2026-05 新论文，部分工具会误判为"伪造未来日期"，这是工具知识截止的工件，**不是 proposal 的错** | arXiv API + 官方 repo + 主页 + HF 四源 **[已核实]** |
| C2 | "每帧含 1 camera token + **16** register tokens"（第 47–51 行） | ⚠️ **register 数是可配置项，非固定 16**。代码应从 config/ckpt **实读** register 数，不要写死 `17 = 1+16` | 论文/代码均未给固定值 **[已核实]** |
| C3 | "registers 携带场景级、几何相关**甚至语义相关**的信息……可用于 VLA 和语言对齐"（第 11 行） | ⚠️ 改写为 **"论文宣称"**：公开材料**无 VLA benchmark/消融**；且语言对齐**可能来自独立的 `text_alignment_embedding`（`enable_alignment=True`）而非 register 本身** | 摘要确有此说，但实验未公开；相关论断投票 1-2 存疑 **[论文宣称]** |
| C4 | "直接输出 camera/register tokens……复用 readout 思路"（第 33、39–67 行） | ⚠️ 补一句：**VGGT-Ω/VGGT 不原生暴露"场景级 readout embedding"**（原版 register 输出被丢弃），`RegisterReadoutHead` 是**必须自训练**的新模块，而非现成功能 | 原版 VGGT register 输出被显式 discarded **[已核实]** |
| C5 | 把 Skeletal Sets / coreset 当作直接先例（第 463–465、730 行） | ⚠️ **引用对，但要标注边界**：① coreset 是"子集上**重训**模型仍有竞争力"，本方案是"**冻结**模型 embedding 逼近"，范式不同；② Skeletal Sets **保留全部 N 张**再注册，本方案**只用 K 张**重建，范围不同 | 形式/范围错配 **[已核实]** |

> 另：不要照搬"GradMatch 一致优于其他数据选择方法"这类强结论 —— 该论断在核查中被**否决（0-3）**。

---

## 三、方法层面的软肋与建议（按优先级）

### 🔴 P0 — 立项门槛：embedding 距离 ≠ 重建质量（必须先验证）

- **问题**：方案的全部价值押在"register embedding 距离小 ⇒ 3DGS 质量好"上，但这**无任何理论或文献保证**。Sener-Savarese 的 coreset 界只约束**特征几何覆盖半径**、明确**不保证任何下游指标**；Skeletal Sets / FisherRF 也都用**几何/信息论**准则而非 embedding 距离。全局 embedding 接近 ≠ 局部细节/几何好（proposal 11.3 自己也承认）。
- **建议**：把 proposal 第 9.2 节（阶段 1）**提升为 hard GATE**：先在小规模数据上画 **register cos-sim ↔ 3DGS PSNR 散点 / Spearman ρ**；**ρ ≥ 0.5 且方向稳定才进入 selector 训练**，否则先加几何辅助 loss 或重定目标。**这是花钱训练前最该做的一件事。**

### 🟡 P1 — 可微选择选型太旧

- **问题**：第 8 节只用 straight-through top-K / Gumbel-Top-k，漏了更贴合"选图像子集"的现代方法。
- **建议**：优先评估 **Reparameterizable Subset Sampling**（[1901.10517](https://arxiv.org/abs/1901.10517), IJCAI'19，直接可微子集采样）与 **Differentiable Patch Selection / perturbed top-K**（[2104.03059](https://arxiv.org/abs/2104.03059), CVPR'21，"选视觉单元子集"范式），**SOFT top-K(OT)**（[2002.06504](https://arxiv.org/abs/2002.06504)）作消融。并正视**前向不可分**：子集真正送进冻结 VGGT-Ω 是离散的，建议先用 **soft-mask 暖启动**再切真实离散，监控 train/eval gap。

### 🟡 P2 — baseline 缺现代视角选择

- **建议**：在第 9.1 节 8 个 baseline 基础上**新增**：**FisherRF**（[2311.17874](https://arxiv.org/abs/2311.17874)，信息增益准则，直接对照"信息论 vs embedding"）、**InstantSplat co-visibility**（[2403.20309](https://arxiv.org/abs/2403.20309)，foundation-model 初始化去冗余）。learned selector 至少要打过 FisherRF 才有说服力。

### 🟡 P3 — readout 可迁移性未验证

- **建议**：落地前**直接读** `vggt_omega/models/heads/text_alignment_head.py`，确认其 readout 思路能否**脱离语言对齐目标**迁移到几何表征蒸馏（这是 proposal 第 67 行设计的可行性边界）。

### 🟡 P4 — 部署语义要先选定

- **建议**：明确二选一并写进方案首段：**(a) 离线数据集压缩**（selector 可用 VGGT register summary，省的是后续 3DGS/VLA 训练成本，**不省 VGGT 推理**）vs **(b) 降低 VGGT 推理成本**（selector 必须用 DINO/CLIP 等便宜特征）。这决定 selector 输入与整个故事的卖点（proposal 11.4 已点出，应前置）。

---

## 四、建议的下一步（Action Items）

- [ ] **(P0)** 跑 Stage 1 相关性实验：固定几个 K，对 random/uniform/k-center 选出的子集，画 `register cos-sim` vs `3DGS PSNR` 散点，算 Spearman ρ。**ρ 不达标就不要进入训练。**
- [ ] **(P3)** 读 `text_alignment_head.py`，确认 readout 能否脱离语言目标；顺带从 config/ckpt 确认 register token 实际数量（修正 C2）。
- [ ] **(P4)** 在方案首段敲定部署语义 (a)/(b)，据此固定 selector 输入特征。
- [ ] **(P2)** 把 FisherRF、InstantSplat co-visibility 纳入 baseline 清单。
- [ ] **(P1)** 选 1–2 种现代可微子集采样（建议 Reparameterizable Subset Sampling + perturbed top-K）做 MVP 对照。
- [ ] **(查新)** 补一轮针对性检索，确认"几何 foundation model 全局 embedding + 可微 selector 选 3D 视角子集"这一组合是否已有等同工作（本轮 WebSearch 故障未穷尽）。
- [ ] 把 proposal 第 11 行的"registers 携带语义信息……可用于 VLA/语言对齐"改写为"论文宣称"口径（修正 C3）。

---

## 五、一句话总结

> **方向对、引用真、自我批判到位；但请先用一个小实验证明"register 距离真的和 3DGS 质量相关"，再决定是否投入训练 —— 这是全案唯一的生死门槛。其余都是可执行的补强。**
