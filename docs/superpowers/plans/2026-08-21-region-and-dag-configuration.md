+# Region and DAG Configuration — Historical Decision

区域与 DAG 的可调参数从 TOML 读取，并在 `env/settings/` 转换为类型化配置；区域生成和 DAG 生成位于 `env/entities/region.py` 与 `env/workload/dag_generator.py`。

随机性现在由实验总 seed 统一派生，不再由此主题配置独立 seed。当前参数以 `config/region.toml`、`config/dag.toml` 和实验配置为准。
