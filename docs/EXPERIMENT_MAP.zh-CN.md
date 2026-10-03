# 实验与论文的对应关系

请先读正文的三个结果小节。下表给出主要图对应的计算入口；同一实验的其他表达保留在补充材料。

| 当前正文图 | 回答的问题 | 主要代码或数据来源 |
|---|---|---|
| 图1 | 输入、三种传递方法和评价如何衔接 | transfer_core_regularized.py、compute_point_diagnostics_v3.py；图为现有结果的流程示例 |
| 图2 | 日尺度更新能否改善小时估计，哪个方法平均误差最低 | predict_transfer_regularized.py、evaluate_transfer_regularized.py、compute_point_diagnostics_v3.py；直接复算用 review_reproduce.py |
| 图3 | 方法对极值尾部的传递差异 | compute_point_diagnostics_v3.py |
| 图4 | 哪些允许的输入特征对应更大的更新收益或方法差异 | complete_submission_checks_wace.py 的 grid 阶段 |
| 图5 | 当跨历时变化与某种假设一致时，额外结构是否有用 | compute_controls_v3.py 的已知分布试验 |
| 图6 | 记录长度如何影响方法误差和估计波动 | compute_controls_v3.py 的样本长度试验 |
| 图7 | 方法差异对设计雨量及超越概率意味着什么 | engineering_design_diagnostics_wace.py |

补充材料中的其他重点包括：bootstrap 区间、阈值敏感性、重现期区段、趋势及去趋势敏感性、站点比较、工程放大系数取舍。对应脚本分别为 bootstrap_joint_v3.py / audit_uncertainty_v3.py、complete_submission_checks_wace.py、complete_trend_sensitivity_wace.py、站点脚本组，以及 evaluate_fixed_uplifts_wace.py。

## 本次包的核验范围

- 四个成员的全部网格和七个历时：从精简参数输入重新生成预测，复算 D、偏差和形态误差等汇总，核对 112 行结果。
- 站点：从包内 AMS 重新做 L-moment GEV 拟合，再做传递和评价，核对 196 行结果。正文相关结论使用六个主站；另一个站保留为探索性比较。
- 正文和补充材料：在独立目录重新编译；图文件逐一核对，论文文字与当前工作稿核对。
- 原始逐小时提取、完整网格 bootstrap 和覆盖率全量试验本次未重新运行；保留相应代码、输入说明及原结果。

这种组织方便审稿人先低成本核验主要结论，再按问题查看具体实验。它不改变论文的方法、参数或结论。
