+# BS Compute Node — Historical Decision

基站作为全局共享的计算节点参与每时隙调度和运行时结算；其参数来自 `config/uav.toml`，运行时队列位于 `env/runtime/`。

当前节点选择与合法性校验由 `methods/` 的组件和 `env/contracts.py` 协作完成。
