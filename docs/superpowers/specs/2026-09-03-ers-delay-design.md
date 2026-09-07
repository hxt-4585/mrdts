+# ERS 时延排序设计（历史决策）

ERS 为全部新 DAG 的子任务给出仅基于时延的优先序，不决定执行位置。计算与依赖通信成本的定义、批量提交语义和稳定排序规则以 `specs/01_system_model.md` 为准。

实现位于 `methods/components/ordering/ers.py`。
