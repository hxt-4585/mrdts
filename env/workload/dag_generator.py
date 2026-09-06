"""DAG 任务生成工具。

本模块仅包含 DAG 任务的生成逻辑，不含归一化、图编码、DAG 化简等与生成无关的内容。

DAG 采用「分层」方式生成：
1. 根据节点数 n 与稠密度参数 rho 确定层数；
2. 将 n 个节点划分到各层，各层节点数受随机性参数 delta 控制；
3. 仅在相邻层之间随机连边，并约束每个节点的出度不超过 max_out。
"""

import math
from dataclasses import dataclass

import numpy as np

from env.settings import DAGConfig


@dataclass
class DAG:
    """单个 DAG 任务的数据结构。

    Attributes:
        node_num: 子任务（节点）数量。
        edges: 依赖边列表，每条边为 (前驱节点, 后继节点)。
        node_features: 节点特征矩阵，shape=(node_num, 2)，
                       第 0 列为输入数据量 (KB)，第 1 列为计算量 (CPU cycles)。
        edge_features: 边特征字典，键为 (u, v)，值为前驱 u 传给后继 v 的中间结果数据量 (KB)。
    """

    node_num: int
    edges: list
    node_features: np.ndarray
    edge_features: dict


class DAGGenerator:
    """DAG 任务生成器。"""

    def __init__(self, config=None, *, rng, py_rng):
        # 允许传入自定义配置，便于后续按区域差异调整 DAG 复杂度
        self.config = config if config is not None else DAGConfig.default()
        # 随机流由实验入口注入，生成器不读取或设置种子。
        self._np_rng = rng
        self._py_rng = py_rng

    # ------------------------------ DAG 拓扑生成 ------------------------------ #
    def _generate_layers(self, n, rho, delta):
        """将 n 个节点划分到若干层，返回各层节点数列表。

        - 层数由 rho 控制：层数 = floor(sqrt(n) / rho)，至少 1 层，且不超过 n。
        - 各层节点数服从均值为 n / 层数、标准差为 delta 的正态分布。
        - 划分后修正各层节点数，保证总数为 n 且每层至少 1 个节点。
        """
        # 层数（限制不超过节点数，避免出现空层导致修正死循环）
        length = max(1, min(n, math.floor(math.sqrt(n) / rho)))
        mean = n / length

        # 各层节点数（至少 1 个节点）
        sizes = [max(1, round(s)) for s in self._np_rng.normal(mean, delta, length)]

        # 修正节点总数，使 sum(sizes) == n
        diff = n - sum(sizes)
        idx = 0
        while diff > 0:
            # 总数偏少：逐个补足
            sizes[idx] += 1
            diff -= 1
            idx = (idx + 1) % length
        while diff < 0:
            # 总数偏多：仅减少仍大于 1 的层，避免出现空层
            if sizes[idx] > 1:
                sizes[idx] -= 1
                diff += 1
            idx = (idx + 1) % length

        return sizes

    def _generate_edges(self, sizes, max_out):
        """根据分层结果生成相邻层之间的依赖边，返回边列表。

        每个节点随机选择 [1, max_out] 个后继节点，后继节点均来自下一层，
        从而保证生成的是有向无环图（DAG）。
        """
        # 生成各层的节点编号
        layers = []
        current = 0
        for size in sizes:
            layers.append(list(range(current, current + size)))
            current += size

        edges = []
        for i in range(len(layers) - 1):
            next_layer = layers[i + 1]
            for u in layers[i]:
                # 出度限制在 [1, min(max_out, 下一层节点数)] 之间
                out_degree = self._py_rng.randint(1, min(max_out, len(next_layer)))
                for v in self._py_rng.sample(next_layer, out_degree):
                    edges.append((u, v))

        return edges

    def generate_topology(self):
        """生成单个 DAG 的拓扑结构。

        Returns:
            (edges, node_num)：依赖边列表与节点数量。
        """
        cfg = self.config
        sizes = self._generate_layers(cfg.n, cfg.rho, cfg.delta)
        edges = self._generate_edges(sizes, cfg.max_out)
        return edges, cfg.n

    # ------------------------------ 节点与边特征生成 ------------------------------ #
    def generate_node_features(self, node_num):
        """生成节点特征：输入数据量与计算量。

        Args:
            node_num: 节点数量。

        Returns:
            shape=(node_num, 2) 的特征矩阵，第 0 列为输入数据量 (KB)，
            第 1 列为计算量 (CPU cycles)。
        """
        cfg = self.config
        # 输入数据量 (KB)：整数，[low, high] 闭区间均匀随机
        input_data = self._np_rng.integers(
            cfg.input_data_range[0], cfg.input_data_range[1] + 1, size=(node_num, 1)
        )
        # 计算量 (CPU cycles)：连续均匀随机
        cpu_cycles = self._np_rng.uniform(
            cfg.cpu_range[0], cfg.cpu_range[1], size=(node_num, 1)
        )
        return np.hstack((input_data, cpu_cycles))

    def generate_edge_features(self, edges):
        """生成边特征：前驱节点传给后继节点的中间结果数据量。

        Args:
            edges: 依赖边列表。

        Returns:
            以 (u, v) 为键、中间结果数据量 (KB) 为值的字典。
        """
        cfg = self.config
        edge_features = {}
        for u, v in edges:
            edge_features[(u, v)] = self._py_rng.randint(
                cfg.intermediate_data_range[0], cfg.intermediate_data_range[1]
            )
        return edge_features

    # ------------------------------ 完整 DAG 生成 ------------------------------ #
    def generate_single_dag(self):
        """生成一个完整的 DAG 任务（拓扑 + 节点特征 + 边特征）。"""
        edges, node_num = self.generate_topology()
        node_features = self.generate_node_features(node_num)
        edge_features = self.generate_edge_features(edges)
        return DAG(
            node_num=node_num,
            edges=edges,
            node_features=node_features,
            edge_features=edge_features,
        )

    def generate(self, num_dags):
        """批量生成 num_dags 个 DAG 任务。

        Args:
            num_dags: 需要生成的 DAG 数量。

        Returns:
            长度为 num_dags 的 DAG 对象列表。
        """
        return [self.generate_single_dag() for _ in range(num_dags)]
