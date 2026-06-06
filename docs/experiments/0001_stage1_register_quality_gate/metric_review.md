# 指标复盘：register token 应该和什么比较

## 背景

上一轮 `register_mean_cosine` 和 PSNR/SSIM/LPIPS 的 scene 内相关性偏弱。这不一定说明 register token 没有用，更可能说明渲染质量指标主要衡量 appearance fidelity、曝光/锐度/像素对齐，而 register token 的目标更接近三维结构、相机关系、深度和场景级空间理解。

因此后续 Stage 1 不应只用 PSNR/SSIM/LPIPS 做质量门。更合理的方向是引入几何指标，并继续坚持 scene 内比较：对同一个 scene 的 5 个 random、1 个 uniform，以及 full-train(non-test) reference 做相似度和质量排序/相关性，不把不同 scene 的同名 random seed 直接混成一个判断统计。

## 候选指标评价

| 指标 | 能否应用 | 评价 | 主要风险 |
|---|---|---|---|
| Point-cloud precision / recall / F-score@tau | 推荐优先尝试 | 这是 3D reconstruction benchmark 的常用几何指标。Precision 衡量重建点到 GT 的准确性，Recall 衡量 GT 被覆盖的完整性，F-score 用阈值 tau 下的调和平均约束二者，适合检验 subset 是否保留了 scene geometry。 | 需要 GT 或伪 GT；tau 对尺度敏感；3DGS 的 raw Gaussian centers 不一定等价于表面点，最好从 depth fusion/mesh/filtered splats 采样。 |
| Accuracy / Completeness directional distance | 推荐优先尝试 | 可视为点云一阶最近邻距离的两个方向：recon -> GT 是 accuracy，GT -> recon 是 completeness。它们比单一渲染指标更能拆分“准但稀疏”和“全但漂浮”的失败模式。 | 对裁剪范围、点密度、外点过滤很敏感；需要统一 crop/alignment。 |
| Chamfer-L1 / Chamfer-L2 | 推荐优先尝试 | 对两组点云做双向最近邻距离；L1 较稳，L2 对离群点惩罚更重。可作为 continuous geometry score，和 F-score@tau 互补。 | 容易被点密度和 floaters 影响；L2 可能被少量异常点主导；仍需要 GT/伪 GT。 |
| Pose ATE / RPE | 可作为辅助指标 | register token 与相机/多视角几何关系有关；如果 subset 导致 pose drift，ATE/RPE 会比 PSNR 更直接。可对比 full COLMAP 或 full VGGT/FastGS reference pose。 | 当前 FastGS 使用固定 COLMAP source 时，训练本身不估 pose；需要明确评估的是 VGGT/Omega pose 还是重建后 pose。 |
| Depth / normal / point-map consistency | 推荐作为 VGGT-native proxy | VGGT/VGGT-OMEGA 本身预测 depth、point maps、camera 等 3D attributes。对 subset 输出和 full-train reference 做 depth/normal/point-map 一致性，可能比 3DGS rendering 更贴近 register token 的训练语义。 | 需要重新缓存 `--include-depth --include-pose`；full reference 仍是伪真值；动态/无界场景的尺度和遮挡处理要小心。 |
| Multi-view reprojection / track consistency | 推荐作为无 GT proxy | 无需 GT mesh。用 subset 重建的 depth/points 在 held-out 或 co-visible views 中重投影，统计 reprojection error、有效覆盖率、track consistency。它直接衡量三维理解和跨视角一致性。 | 工程量较大；需要可靠 visibility、depth rendering、相机模型和遮挡处理。 |
| Full reconstruction as pseudo-GT | 可用，适合当前数据 | 对 mipnerf360/db 这类没有公开 GT geometry 的场景，可以把 full-train(non-test) reconstruction 当伪真值，在同一 scene 内比较 subset。它比跨 scene 平均更合理，也能保留当前 FastGS pipeline。 | 伪 GT 会继承 full reconstruction 的偏差；如果 full run 自身有 floaters 或几何错误，会把错误当标准；只能作为 relative proxy，不能当绝对几何真值。 |
| No-reference NVS quality metrics | 可作为补充，不优先 | NeRF-NQA 这类无参考 NVS quality 指标试图评估 dense view 的 spatial/angular quality，可在缺 GT view 时补充渲染侧证据。 | 仍偏 perceptual/NVS，不一定对应三维理解；实现和校准成本高于直接几何 proxy。 |
| OpenVLA / VLA downstream success rate | 最强外部验证，但不适合作为当前第一层 gate | VGGT-OMEGA 论文明确把 registers 描述为 scene information carrier，并报告其可提升 VLA/语言对齐。把 register tokens 接入 OpenVLA/OFT 后看 LIBERO/SimplerEnv success rate，能检验“token 对下游空间任务是否有用”。 | 成本很高；需要 robot task data、policy fine-tuning 或 token-injection protocol；和当前 3DGS subset selection 的数据域不一致。建议作为 Stage 2/3 外部效度实验。 |

## 有无真值时的实验口径

有 GT geometry 时：

- 优先使用 GT point cloud / mesh。
- 对每个 scene 和 method 输出 filtered point cloud 或 fused depth point cloud。
- 统一坐标系、scale、crop 和采样密度。
- 计算 accuracy、completeness、Chamfer-L1/L2、F-score@tau。
- 在同一 scene 内比较 5 random + 1 uniform 的 `register_mean_cosine` 或 readout similarity 与这些几何指标的 Spearman/Pearson。

没有 GT geometry 时：

- 使用 full-train(non-test) reconstruction 作为 pseudo-GT，只做 scene 内相对比较。
- pseudo-GT 推荐保留两套：
  - FastGS full-train geometry：贴近最终重建 pipeline。
  - VGGT-Omega full-train depth/point-map：贴近 register token 的原生几何语义。
- 所有结论标注为 pseudo-GT，不和真值指标混报。
- 如果 full reconstruction 质量明显失败，该 scene 不能用于几何 proxy gate。

## 当前 Stage 1 建议

短期可执行：

1. 对 13 个 scene 跑 full-train(non-test) FastGS `images_4` reconstruction，作为 pseudo-GT baseline。
2. 从 full 和 78 个 subset run 提取可比较几何：
   - 首选：rendered depth fusion 或 mesh/surface samples。
   - 次选：过滤 opacity/size 后的 Gaussian centers，但要明确它不是表面真值。
3. 计算 per-scene F-score@tau、accuracy、completeness、Chamfer-L1/L2。
4. 重新做 scene 内相关性：6 个候选 subset 的 token similarity vs geometry metric。
5. 如果 mipnerf360/db 没有 GT，则只以 pseudo-GT 报告；若能拿到 Tanks and Temples training `Truck` GT 或后续加入 DTU，再用真 GT 做 sanity calibration。

中期建议：

- 重新跑 VGGT cache，开启 `--include-depth --include-pose`，把 subset VGGT geometry 与 full-train VGGT geometry 比较。
- 训练或校准 readout head 后，用同样的 geometry metrics 复测，而不是只看 mean-pooled register tokens。
- 补 feature/register k-center 后，检查 geometry metrics 是否能区分 random、uniform、k-center。

长期建议：

- 设计 VLA 下游实验作为外部效度：把 frozen VGGT-Omega registers 接入 OpenVLA/OFT，报告 LIBERO/SimplerEnv success rate、action L1/token accuracy 或 rollout success。这个指标最接近“register 是否帮助空间决策”，但不应阻塞当前几何 gate。

## 参考资料

- VGGT-Ω project page: https://vggt-omega.github.io/
- VGGT paper: https://arxiv.org/abs/2503.11651
- Tanks and Temples benchmark: https://tanksandtemples.org/
- Tanks and Temples download/GT notes: https://tanksandtemples.org/download/
- Middlebury MVS evaluation overview: https://www.microsoft.com/en-us/research/publication/a-comparison-and-evaluation-of-multi-view-stereo-reconstruction-algorithms/
- Learning Local Displacements for Point Cloud Completion, examples of Chamfer-L1/L2 and F-Score@1%: https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_Local_Displacements_for_Point_Cloud_Completion_CVPR_2022_paper.pdf
- Mip-NeRF 360 paper: https://arxiv.org/abs/2111.12077
- NeRF-NQA: https://arxiv.org/abs/2412.08029
- OpenVLA repository and LIBERO success-rate protocol: https://github.com/openvla/openvla
