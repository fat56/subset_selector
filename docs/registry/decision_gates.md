# Decision Gates

## G1: Register Quality Correlation

- Stage: `stage1`
- Question: register/readout embedding similarity 是否能预测 20% sparse-view FastGS 质量？
- Primary statistic: Spearman rho between `register_cosine_similarity` and `psnr`
- Suggested pass: `rho >= 0.5` and positive direction stable across methods/scenes under the 20% ratio budget
- Fail action: do not train selector; revise readout/proxy or add geometry auxiliary objectives

## G2: Fixed-K Selector Beats Strong Baselines

- Stage: `stage2`
- Question: learned selector 在相同 K 下是否优于强 baseline？
- Must compare: random, uniform, feature k-center, register k-center, FisherRF or available information-gain proxy
- Suggested pass: improves reconstruction metrics without sacrificing stability

## G3: Useful Pareto Frontier

- Stage: `stage3`
- Question: 是否能在给定质量降幅内显著减少 K？
- Suggested pass examples: PSNR drop <= 1 dB, SSIM drop <= 0.02, LPIPS increase <= 0.02
