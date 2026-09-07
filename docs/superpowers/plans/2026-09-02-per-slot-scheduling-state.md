+# Per-Slot Scheduling State — Historical Decision

调度状态和资源队列按 slot 隔离，不为后续 slot 保留未完成任务。飞行位置可以在 episode 内连续演化，但每个 episode 从相同初始场景重建。

相关生命周期逻辑位于 `env/simulator.py` 和 `env/runtime/`。
