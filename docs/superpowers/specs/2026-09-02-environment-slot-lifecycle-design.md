+# 环境时隙生命周期（历史决策）

每个 slot 使用独立的通信和计算运行时，截止时结算完成或失败；未完成 DAG 不跨 slot 继续执行。实现位于 `env/simulator.py` 与 `env/runtime/`。
