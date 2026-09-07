+# Region and DAG Configuration Design — Historical Decision

区域和 DAG 参数通过 `env/settings/` 读取为类型化配置；区域和 DAG 的生成实现分别位于 `env/entities/region.py` 与 `env/workload/dag_generator.py`。

随机种子不属于这些公共配置，统一由实验层提供随机流。
