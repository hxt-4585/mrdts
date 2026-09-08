# 实验产物

运行目录：`<experiment>/runs/<solution_ordering_flight_scheduling>/seed_<seed>/<UTC timestamp + ID>/`。

- `config.json`：最终实验/方法配置及每回合完整场景参数；新运行只在 `experiment.seed` 保存总种子，模块配置不含种子。采用 JSON 以无损保存嵌套配置。
- `metadata.json`：模式、状态、代码版本、工作区状态、种子与时间。
- `metrics.csv`：逐时隙公共指标。
- `summary.json`：按 DAG 数聚合的指标，成功时延与失败惩罚分开记录。
- `figures/`：按需生成图表；PPO 训练曲线在 `figures/training/`。
- `training/`：PPO 的 epoch、update、validation CSV、进度与最终 test 结果。
- `checkpoints/`：PPO 的最新完整 epoch、验证选优模型与末态。

汇总输出位于 `<experiment>/analysis/comparison.csv`，保留每个已完成的评估运行和种子，排除训练运行，不自动合并不兼容的实验。
只提交此说明文件；大量运行产物不纳入 Git。单次汇总/绘图会更新该实验的派生产物，原始运行目录不会覆盖。
