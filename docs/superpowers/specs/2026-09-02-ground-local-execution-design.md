+# Ground-Device Local Execution Design — Historical Decision

任务所属 Ground 是合法的本地执行候选，拥有单核且不计计算能耗；它和 UAV、BS 共享当前 slot 的运行时生命周期。

实现以 `env/runtime/`、`env/contracts.py` 和调度组件为准。
