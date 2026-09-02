# 每时隙独立调度状态设计

## 背景

每个时隙先完成固定 0.5 秒的 Member UAV 飞行，再在飞行后的静止拓扑上执行最长 1 秒的 DAG 调度。调度状态不延续到下一时隙；只有 UAV 位置、能量、区域归属和能量虚拟队列等物理状态可以跨时隙保留。

当前代码同时使用 `MemberUAV.core_available_at` 和 `SchedulingRuntime` 内的 `ServerState` 表示核心占用，形成两个可能不一致的计算状态源。

## 决定

- 每个时隙创建一个全新的 `SchedulingRuntime`。
- 信道队列、服务器队列、核心占用和 DAG 运行时仅在当前时隙有效。
- 时隙调度窗口结束后丢弃整个 `SchedulingRuntime`；未完成 DAG 由后续强化学习环境记为失败，不带入下一时隙。
- `SchedulingRuntime` 中的 `ServerState` 是时隙内计算占用的唯一权威状态。
- `MemberUAV` 只保存 `core_counts` 和 `core_frequencies` 等静态计算能力，不保存 `core_available_at`。
- 删除 `MemberUAV.schedule_computation()`、`Environment.estimate_finish_time()` 和 `Environment.reserve_computation()`，避免旧接口重新建立第二套状态。
- 固定飞行阶段不进入调度离散事件时间线；调度时延从飞行完成后的调度起点计算。

## 保留范围

- 本次不实现同一时隙多 DAG 的统一 ERS；该功能在下一阶段单独完成。
- 本次不实现覆盖半径、越界动作或空区域惩罚；这些约束由后续强化学习环境处理。

  后续决定（2026-09-02）：当前不启用通信覆盖半径约束，用户关联采用“同区域最近 Member”，本区域 Member 执行候选也不按覆盖半径筛选。`coverage_radius` 配置及其字段暂时保留，覆盖半径检查不属于正式强化学习环境必须补齐的当前模型要求。
- 事件运行时仍可推进到任意当前时隙内的绝对时间，但不承诺跨时隙复用。

后续实现（2026-09-02）：`Environment.begin_slot/end_slot` 已负责正式时隙开始和结束；运行时绑定窗口并在截止时自动关闭，关闭后禁止继续推进、提交或更新拓扑。未完成 DAG 标记失败，环境释放旧运行时引用。详见 `2026-09-02-environment-slot-lifecycle-design.md`。

## 验证

- 测试确认 `MemberUAV` 不再暴露 `core_available_at` 和 `schedule_computation()`。
- 测试确认 `Environment` 不再暴露旧的计算估算与预留接口。
- 现有 `SchedulingRuntime`、`ServerState` 和完整测试套件继续通过。
- 全库检索确认文档不再声称调度状态或未完成 DAG 跨时隙保留，也不再要求把固定飞行时间写入调度事件时间线。
