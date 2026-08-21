# BS 计算节点设计

## 目标

在 `MemberUAV` 的向量化计算状态中追加一个固定 BS 行，以保留统一矩阵运算，同时明确 BS 不参与 Member 的飞行和迁移行为。

## 状态契约

- `MemberUAV.positions` 和 `MemberUAV.region_ids` 的长度为 `member_uav_count + 1`。
- 索引 `0 .. member_uav_count - 1` 是 Member；`bs_index == member_uav_count` 是 BS。
- Member 的区域编号为 `1..R`；BS 的区域编号为 `0`。
- BS 固定在正方形仿真区域中心，位置为 `(side_length / 2, side_length / 2, 25 m)`。
- `member_uav_count` 表示可移动 Member 数量；`num_uavs` 表示包含 BS 的计算节点总数。
- 飞行动作输入仍是 `(member_uav_count, 2)`，仅更新 Member 行。

## 计算资源

- 每个 Member 具有 2 个核心，每核心频率为 `10e9` cycles/s。
- BS 具有 4 个核心，每核心频率为 `12e9` cycles/s。
- `core_counts` 的形状为 `(member_uav_count + 1,)`；`core_frequencies` 的形状为 `(member_uav_count + 1, 4)`，不存在的核心频率填 `0`。

## 范围

本变更只建立 BS 行、配置和计算核心矩阵；不实现 Member–BS 概率 LoS/NLoS 信道、调度或能耗模型。
