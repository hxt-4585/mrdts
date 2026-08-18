"""MRDTS DAG 任务生成配置。

只保留 DAG 任务生成所需的参数，参数取值与符号依据
specs/03_task_dag_model.md 和 specs/02_scenario_parameters.md 第 8 节。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DAGConfig:
    """DAG 任务生成参数。

    符号对应关系（见 specs/03_task_dag_model.md）：
    - N_dag : 每个 DAG 包含的子任务数量；
    - d_max : 单个子任务最多依赖的后继任务数量（最大出度）；
    - rho   : 形状参数（稠密度参数），控制 DAG 深度（层数），
              rho 越大层数越少、DAG 越扁平；
    - delta : 随机性参数（规则性参数），控制各层节点数的波动程度，
              delta 越大各层节点数越不均匀。
    """

    # ---------- DAG 拓扑参数 ----------
    num_subtasks: int = 10                # DAG 节点数 N_dag（子任务数量）
    max_out_degree: int = 2               # 最大出度 d_max
    shape_parameter: float = 1.0          # 形状参数 rho，控制 DAG 深度（层数）
    regularity_parameter: float = 0.5     # 随机性参数 delta，控制各层节点数波动

    # ---------- 子任务参数 ----------
    input_data_size_kb: Tuple[int, int] = (100, 500)      # 子任务输入数据量范围（KB）
    cpu_cycles: Tuple[float, float] = (1e7, 1e8)          # 子任务计算量范围（CPU cycles）
    intermediate_data_size_kb: Tuple[int, int] = (100, 300)  # 前驱中间结果数据量范围（KB）


DEFAULT_DAG_CONFIG = DAGConfig()
