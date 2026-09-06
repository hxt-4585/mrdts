# Random 方案

与 `proposed/` 平级的完整非学习方法，默认组合为 `ers + random + random`。

- 排序：移动后使用现有 ERS，算法不变。
- 飞行：各 Master 对原所属 Member 的 x/y 归一化动作独立从 [-1, 1] 均匀采样。不是固定速度的均匀方向采样。
- 调度：移动并更新用户关联后，每个 Member 沿 ERS 顺序，逐子任务从合法节点集合等概率选择执行位置。集合包含任务自己的地面设备、同区域 Member UAV 和全局 BS。
- 地图越界：沿用现有环境行为，该 Member 本次保持原位置，并统计越界。
- 服务区域迁空：整批动作拒绝并重新采样，最多 128 次；仍不可行则全体保持原位。`flight_resamples` 记录被拒绝批次数，`flight_fallbacks` 记录回退。

这是带上述可行性筛选的随机飞行，不是无约束飞行动作的均匀分布。预览复用 UAV 移动函数并复制位置，不改变真实仿真状态。合法跨区迁移仍然允许。

`experiments/randomness.py` 根据实验配置中的总 seed 和回合编号统一构建随机流，通过 `create_method(config, flight_rng=..., scheduling_rng=...)` 注入方法。方法本身没有 seed 或备用随机源；飞行、调度的随机流相互独立，也独立于区域、用户和 DAG。
Random 不需要训练，通过 `experiments/run.py` 或 `python -m experiments.run` 执行。默认配置为 `config/experiments/random.toml`。
