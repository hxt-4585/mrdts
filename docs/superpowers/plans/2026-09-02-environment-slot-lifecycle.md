+# Environment Slot Lifecycle — Historical Decision

每个 slot 建立新的通信与计算运行时，在截止窗口结算；未完成 DAG 失败，剩余工作不跨 slot 保留。

实现位于 `env/simulator.py` 与 `env/runtime/`，当前调度窗口以 `config/scheduling.toml` 为准。
