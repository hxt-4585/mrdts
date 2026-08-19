"""DAG 任务生成的参数配置。

本文件仅保留 DAG 任务生成相关的参数，取值与 specs 文档保持一致：
- specs/02_scenario_parameters.md 第 8 节「DAG 任务设置」
- specs/03_task_dag_model.md
"""


class DAGConfig:
    """DAG 任务生成参数配置。

    Attributes:
        n: 每个 DAG 包含的子任务（节点）数量。
        max_out: 单个子任务的最大出度（后继子任务数量）。
        rho: 稠密度参数，控制 DAG 层数（rho 越大，层数越少、DAG 越浅）。
        delta: 随机性参数，控制各层子任务数量的波动程度。
        input_data_range: 子任务输入数据量范围 (KB)，均匀随机生成。
        cpu_range: 子任务计算量范围 (CPU cycles)，均匀随机生成。
        intermediate_data_range: 前驱子任务中间结果数据量范围 (KB)，均匀随机生成。
    """

    def __init__(self):
        # ---------------- DAG 拓扑参数 ----------------
        self.n = 10                          # 每个 DAG 的子任务数
        self.max_out = 2                     # 最大出度
        self.rho = 1.0                       # 稠密度参数（控制层数）
        self.delta = 0.5                     # 随机性参数（控制各层节点数波动）

        # ---------------- 子任务参数 ----------------
        self.input_data_range = (100, 500)   # 输入数据量 (KB)
        self.cpu_range = (1e7, 1e8)          # 计算量 (CPU cycles)
        self.intermediate_data_range = (100, 300)  # 中间结果数据量 (KB)
