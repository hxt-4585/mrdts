"""MRDTS DAG 任务生成器。

本模块只负责 DAG 任务的生成，不含任何与图神经网络、特征归一化、
DAG 化简等无关代码（相关代码已在重构中移除，原模块中的
build_dag_batch / DAG_reduction / server_state_generator 等均已删除）。

生成流程：
1. 按层生成 DAG 拓扑：层数由形状参数 rho 控制，各层节点数从正态分布
   采样（由随机性参数 delta 控制波动），并将总和修正为 num_subtasks；
2. 连通性可行性修正：保证相邻两层满足"第 i 层节点数 <= max_out *
   第 i-1 层节点数"，使后续连边不出现顾此失彼的情况；
3. 逐层生成依赖边：先通过匹配边保证每个非首层节点至少有一个前驱，
   再通过补充边让每个非末层节点在最大出度内补足后继，
   使 DAG 连通成"一个任务"，且出度严格不超过 max_out；
4. 为每个子任务生成输入数据量、计算量，为每条依赖边生成中间结果数据量。

随机性统一使用 numpy 的 Generator（单一随机流），不固定随机种子；
如需复现实验结果，可在外部统一设置 numpy 全局随机种子。
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.config import DAGConfig, DEFAULT_DAG_CONFIG


@dataclass
class DAGTopology:
    """DAG 拓扑结构。"""

    edges: List[Tuple[int, int]]    # 依赖边列表，元素为 (前驱节点, 后继节点)
    node_num: int                   # 节点（子任务）总数
    layer_sizes: List[int]          # 各层节点数，按层顺序排列
    in_degrees: List[int]           # 各节点入度（前驱数量）
    out_degrees: List[int]          # 各节点出度（后继数量）


@dataclass
class TaskDAG:
    """一个完整的 DAG 任务：拓扑结构 + 节点/边特征。"""

    topology: DAGTopology                          # 拓扑结构
    node_data_sizes: np.ndarray                    # 各子任务输入数据量（KB），形状 (N, 1)
    node_cpu_cycles: np.ndarray                    # 各子任务计算量（CPU cycles），形状 (N, 1)
    edge_data_sizes: Dict[Tuple[int, int], int]    # 各依赖边携带的中间结果数据量（KB）
    total_data_size: float                         # DAG 总输入数据量（KB）
    total_cpu_cycles: float                        # DAG 总计算量（cycles）


class DAGGenerator:
    """DAG 任务生成器。

    用法示例：
        generator = DAGGenerator()             # 使用默认配置
        dag  = generator.generate_dag()        # 生成一个 DAG 任务
        dags = generator.generate_dags(10)     # 批量生成 10 个 DAG 任务

    Args:
        config: DAG 生成参数，默认为 DEFAULT_DAG_CONFIG。
    """

    def __init__(self, config: Optional[DAGConfig] = None):
        self.config = config or DEFAULT_DAG_CONFIG
        # 统一使用单一随机流，避免 np.random 与 random 两个模块混用
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # DAG 拓扑生成
    # ------------------------------------------------------------------ #
    def generate_topology(self) -> DAGTopology:
        """生成 DAG 拓扑结构。

        算法说明（继承自原 JTPDTS 代码的分层思想，并修正其中的问题）：

        1. 层数 length = floor(sqrt(N_dag) / rho)，并限制在 [1, N_dag] 之间。
           （原代码未限制上限：当 rho 很小时层数可能超过节点数，
            使"每层至少 1 个节点"与"总数为 N_dag"无法同时满足，
            修正循环存在死循环风险。）
        2. 各层节点数从正态分布 N(N_dag / length, delta^2) 采样后向上取整
           （至少为 1），再随机增减某些层的节点数，将总和修正为 N_dag。
           由于 length <= N_dag，不可能出现"每层均为 1 而总和仍超限"，
           修正循环必然终止。
        2.5 连通性可行性修正：要求相邻两层满足"第 i 层节点数 <= max_out *
            第 i-1 层节点数"，否则把多余节点挪到第 0 层，使后续连边能
            同时满足"每个非首层节点至少有一个前驱"与"出度不超过
            max_out"（详见代码内注释）。
        3. 逐层生成依赖边，由构造保证以下性质（原逻辑会以较高概率生成
           入度为 0 的中间层节点，使 DAG 分裂成多个互不相连的子图）：
           - 每个节点的出度不超过 max_out_degree；
           - 每个非末层节点至少有一个后继（出度 >= 1）；
           - 每个非首层节点至少有一个前驱（入度 >= 1），
             整个 DAG 连通成"一个任务"。
           具体分两步：
           3.1 匹配边：将下一层节点按环状映射到当前层节点，保证每个
               下一层节点都有前驱；当前层节点出度已满时，优先换用
               当前层仍有出度余量的节点，其次允许跨层从更早的层选取
               （可行性修正已保证此兜底分支实际上不会触发）；
           3.2 补充边：每个当前层节点在不超过 max_out 的前提下，
               随机补充后继，使出度达到上限。
           所有边都从第 i 层（或更早的层）指向第 i+1 层，因此天然无环。
        """
        n = self.config.num_subtasks
        max_out = self.config.max_out_degree
        rho = self.config.shape_parameter
        delta = self.config.regularity_parameter

        # 1. 计算层数（DAG 深度），至少 1 层、至多 N_dag 层
        length = min(n, max(1, math.floor(math.sqrt(n) / rho)))

        # 2. 生成各层节点数：正态采样 + 取整，再修正使总和等于 n
        mean_value = n / length
        random_num = self._rng.normal(loc=mean_value, scale=delta, size=length)
        layer_sizes = [max(1, math.ceil(r)) for r in random_num]
        diff = n - sum(layer_sizes)
        while diff != 0:
            idx = int(self._rng.integers(0, length))
            if diff > 0:
                layer_sizes[idx] += 1
                diff -= 1
            elif layer_sizes[idx] > 1:
                layer_sizes[idx] -= 1
                diff += 1

        # 2.5 连通性可行性修正：连边时，一个节点能提供的出边只受其所在层
        #     约束，而更早的层在上一轮连边中已被补充边填满到 max_out，
        #     无法再借出容量。因此要保证"每个非首层节点至少有一个前驱"
        #     与"出度不超过 max_out"能同时满足，只需（也必须）要求相邻层
        #     满足：第 i 层节点数 <= max_out * 第 i-1 层节点数。
        #     若不满足（例如第 1 层节点数超过 max_out 倍的第 0 层节点数，
        #     第 0 层无法为第 1 层所有节点供边），则将多余节点从该层挪到
        #     第 0 层，保持总节点数与各层非空不变；该修正只影响少数极端
        #     层分布。
        for i in range(1, length):
            while layer_sizes[i] > max_out * layer_sizes[i - 1]:
                layer_sizes[i] -= 1
                layer_sizes[0] += 1

        # 层内节点编号：按层连续编号（在可行性修正之后进行）
        layers: List[List[int]] = []
        current = 0
        for size in layer_sizes:
            layers.append(list(range(current, current + size)))
            current += size

        edges: List[Tuple[int, int]] = []
        edge_set: set = set()
        in_degrees = [0] * n
        out_degrees = [0] * n

        def _add_edge(u: int, v: int) -> None:
            """添加一条依赖边并更新两端节点的度数。"""
            edges.append((u, v))
            edge_set.add((u, v))
            in_degrees[v] += 1
            out_degrees[u] += 1

        # 3. 逐层生成依赖边
        for i in range(length - 1):
            cur_level, next_level = layers[i], layers[i + 1]

            # 3.1 匹配边：保证 next_level 中每个节点至少有一个前驱。
            #     下一层节点按环状依次映射到当前层节点，尽量均匀分担出度。
            cur_shuffled = list(cur_level)
            next_shuffled = list(next_level)
            self._rng.shuffle(cur_shuffled)
            self._rng.shuffle(next_shuffled)
            for j, v in enumerate(next_shuffled):
                u = cur_shuffled[j % len(cur_shuffled)]
                if out_degrees[u] >= max_out:
                    # 该节点出度已满：优先换用当前层仍有出度余量的节点
                    candidates = [x for x in cur_level if out_degrees[x] < max_out]
                    if candidates:
                        u = int(self._rng.choice(candidates))
                    else:
                        # 当前层全部出度已满：允许跨层，从更早的层寻找余量
                        earlier = [node for layer in layers[:i] for node in layer]
                        candidates = [x for x in earlier if out_degrees[x] < max_out]
                        if candidates:
                            u = int(self._rng.choice(candidates))
                        else:
                            # 更早的层也已全部出度已满：此分支仅作为防御性
                            # 兜底保留——2.5 节的可行性修正已保证相邻层容量
                            # 足够（第 i+1 层节点数 <= max_out * 第 i 层
                            # 节点数），正常情况下不会执行到这里。
                            pool = earlier or cur_level
                            u = min(pool, key=lambda x: out_degrees[x])
                if (u, v) not in edge_set:
                    _add_edge(u, v)

            # 3.2 补充边：每个当前层节点在不超过 max_out 的前提下随机补充后继
            for u in cur_level:
                unconnected = [v for v in next_level if (u, v) not in edge_set]
                self._rng.shuffle(unconnected)
                while out_degrees[u] < max_out and unconnected:
                    _add_edge(u, unconnected.pop())

        return DAGTopology(
            edges=edges,
            node_num=n,
            layer_sizes=layer_sizes,
            in_degrees=in_degrees,
            out_degrees=out_degrees,
        )

    # ------------------------------------------------------------------ #
    # 任务特征生成
    # ------------------------------------------------------------------ #
    def generate_dag(self) -> TaskDAG:
        """生成一个完整的 DAG 任务（拓扑结构 + 节点/边特征）。

        特征取值依据 specs/03_task_dag_model.md：
        - 子任务输入数据量：均匀随机生成于 [100, 500] KB；
        - 子任务计算量：均匀随机生成于 [1e7, 1e8] CPU cycles；
        - 前驱中间结果数据量：均匀随机生成于 [100, 300] KB。
        """
        topology = self.generate_topology()
        n = topology.node_num

        # 各子任务输入数据量（KB）
        node_data_sizes = self._rng.integers(
            self.config.input_data_size_kb[0],
            self.config.input_data_size_kb[1] + 1,  # integers 为左闭右开，+1 保持闭区间
            size=(n, 1),
        )
        # 各子任务计算量（CPU cycles）
        node_cpu_cycles = self._rng.uniform(*self.config.cpu_cycles, size=(n, 1))
        # 各依赖边携带的中间结果数据量（KB）
        edge_data_sizes = {
            edge: int(self._rng.integers(
                self.config.intermediate_data_size_kb[0],
                self.config.intermediate_data_size_kb[1] + 1,
            ))
            for edge in topology.edges
        }

        return TaskDAG(
            topology=topology,
            node_data_sizes=node_data_sizes,
            node_cpu_cycles=node_cpu_cycles,
            edge_data_sizes=edge_data_sizes,
            total_data_size=float(np.sum(node_data_sizes)),
            total_cpu_cycles=float(np.sum(node_cpu_cycles)),
        )

    def generate_dags(self, num_dags: int) -> List[TaskDAG]:
        """批量生成多个 DAG 任务。

        Args:
            num_dags: 需要生成的 DAG 任务数量。

        Returns:
            由 num_dags 个 TaskDAG 组成的列表。
        """
        return [self.generate_dag() for _ in range(num_dags)]


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # 示例与自检：验证生成逻辑的正确性
    # ------------------------------------------------------------------ #

    def _self_check(generator: DAGGenerator, num_samples: int = 5000) -> None:
        """批量生成 DAG 并统计各类非法情况。"""
        bad_node_num = 0        # 节点总数不等于配置值
        bad_connectivity = 0    # 存在入度为 0 的非首层节点（DAG 不连通）
        bad_out_degree = 0      # 存在出度超过最大出度的节点
        for _ in range(num_samples):
            t = generator.generate_topology()
            if t.node_num != generator.config.num_subtasks:
                bad_node_num += 1
                continue
            # 非首层节点（编号 >= 首层节点数）入度必须 >= 1
            first_layer_size = t.layer_sizes[0]
            if any(d == 0 for d in t.in_degrees[first_layer_size:]):
                bad_connectivity += 1
            # 出度不得超过最大出度（极端配置下的退化情形除外）
            if any(d > generator.config.max_out_degree for d in t.out_degrees):
                bad_out_degree += 1

        print(f"  节点总数错误: {bad_node_num}")
        print(f"  入度为 0 的非首层节点（不连通）: {bad_connectivity}")
        print(f"  出度超过最大出度: {bad_out_degree}")

    # 示例：生成并打印一个 DAG
    example = DAGGenerator().generate_dag()
    topo = example.topology
    print("===== 单个 DAG 示例（默认参数）=====")
    print(f"层数: {len(topo.layer_sizes)}，各层节点数: {topo.layer_sizes}")
    print(f"边数: {len(topo.edges)}")
    print(f"入度: {topo.in_degrees}")
    print(f"出度: {topo.out_degrees}")
    print(f"总输入数据量: {example.total_data_size:.1f} KB，"
          f"总计算量: {example.total_cpu_cycles:.3e} cycles")

    # 自检：默认参数
    print("\n===== 自检：默认参数（5000 个 DAG）=====")
    _self_check(DAGGenerator())

    # 自检：极端参数（多而窄的层、节点数波动大）
    extreme_cfg = DAGConfig(
        num_subtasks=30, max_out_degree=3,
        shape_parameter=0.3, regularity_parameter=3.0,
    )
    print("\n===== 自检：极端参数（{} 个 DAG）=====".format(extreme_cfg))
    print(f"  配置: {extreme_cfg}")
    _self_check(DAGGenerator(extreme_cfg))
