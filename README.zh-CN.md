# 日尺度到小时尺度设计降雨：实验代码

本仓库对应论文 **Updating hourly design rainfall from daily projections: How much complexity is needed?**。
核心比较为 NC、UCF、HQT、TPS 四种方法，包含 AMS 提取、L-moment GEV 拟合、传递预测、趋势检查、bootstrap、已知分布与样本长度试验、站点验证及工程评价。

## 如何使用

1. 使用 Python 3.12，并安装 `requirements.txt`。
2. 将本地审稿包中的精简数据导入项目：

```text
python tools/prepare_review_data.py --source "你的审稿包/02_experiments"
python reproduce.py --threads 4
```

输出位于 `analysis/reviewer_output`。这一步会复算四成员全部网格的主要误差指标，并从站点 AMS 重新拟合、评价。未来小时数据只用于评价。

仓库只跟踪代码和说明；数据、论文、缓存与计算输出均被排除。完整原始流程需要的数据见 `docs/DATA_REQUIREMENTS.md`。仅需要查看方法时，不必导入数据。

`analysis/script_groups.json` 提供脚本分组。保持原有脚本在同一目录，是为了让它们相互导入时继续正常工作。计算方法和参数未因整理仓库而更改。

详细操作见英文 README，实验与正文图的对应关系见 `docs/EXPERIMENT_MAP.zh-CN.md`。
