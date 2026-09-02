# Member 飞行越界拒绝设计

## 行为

- `MemberUAV.apply_flight_actions(normalized_actions, side_length)` 计算每架 Member 的候选二维位置。
- 候选位置的任一分量不在闭区间 `[0, side_length]` 时，拒绝该 Member 的整次二维移动并保持原位置。
- 合法 Member 正常更新位置；不同 Member 的动作独立判断。
- BS 不移动，也不出现在违规数组中。
- 方法返回 `shape=(member_uav_count,)`、`dtype=bool` 的数组；`True` 表示该 Member 的动作因越界被拒绝。
- 本次不计算具体惩罚数值。后续强化学习奖励逻辑使用返回的违规数组施加惩罚。

## 约束

- `side_length` 必须是有限正数。
- 原有动作 shape、有限值和 `[-1, 1]` 范围校验保持不变。
- `coverage_radius` 不接入用户关联逻辑。
- 不修改 ERS；同一时隙多 DAG 统一 ERS 仍是独立后续工作。

## 验证

- 合法动作更新位置并返回全 `False`。
- 越界动作保持该 Member 原位置并返回对应 `True`。
- 同一批次内合法 Member 仍能移动，BS 始终不动。
- 完整测试套件通过。
