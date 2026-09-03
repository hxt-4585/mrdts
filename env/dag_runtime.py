"""单个活动 DAG 的运行时依赖状态。

本模块不负责卸载决策、传输、计算资源或 UAV 状态；它只追踪哪些子任务
已经满足前驱依赖、何时完成，以及整个 DAG 何时完成。
"""

from dataclasses import dataclass, field
import heapq
from numbers import Integral

import numpy as np

from env.dag_generator import DAG


def topological_order(dag: DAG) -> tuple[int, ...]:
    """校验 DAG 并返回确定性拓扑序；长链不依赖 Python 递归栈。"""
    if isinstance(dag.node_num, bool) or not isinstance(dag.node_num, Integral) or dag.node_num < 0:
        raise ValueError("DAG 节点数必须为非负整数")
    indegree = {node: 0 for node in range(dag.node_num)}
    successors = {node: [] for node in indegree}
    seen = set()
    for parent, child in dag.edges:
        if parent not in indegree or child not in indegree:
            raise ValueError("DAG 边引用了不存在的节点")
        if (parent, child) in seen:
            raise ValueError("DAG 不能包含重复边")
        seen.add((parent, child))
        successors[parent].append(child)
        indegree[child] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        node = heapq.heappop(ready)
        ordered.append(node)
        for child in successors[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(ordered) != dag.node_num:
        raise ValueError("DAG 中存在环")
    return tuple(ordered)


def validate_dag(dag: DAG) -> tuple[int, ...]:
    """检查拓扑和负载；排序及批量提交共用同一数据约束。"""
    order = topological_order(dag)
    features = np.asarray(dag.node_features, dtype=float)
    if features.shape != (dag.node_num, 2):
        raise ValueError("DAG 节点特征必须为 (节点数, 2)")
    if not np.isfinite(features).all() or (features[:, 0] < 0).any() or (features[:, 1] <= 0).any():
        raise ValueError("输入数据量必须有限非负，计算量必须有限为正")
    if set(dag.edge_features) != set(dag.edges):
        raise ValueError("DAG 边数据量必须覆盖全部边")
    sizes = np.asarray(list(dag.edge_features.values()), dtype=float)
    if not np.isfinite(sizes).all() or (sizes < 0).any():
        raise ValueError("边数据量必须为有限非负数")
    return order


@dataclass
class DAGRuntime:
    """一个 DAG 在仿真中的轻量运行时状态。"""

    dag: DAG
    remaining_predecessors: dict[int, int] = field(init=False)
    successors: dict[int, set[int]] = field(init=False)
    finished_at: dict[int, float] = field(default_factory=dict, init=False)
    ready_node_ids: set[int] = field(init=False)
    failed_at: float | None = field(default=None, init=False)

    def __post_init__(self):
        self.remaining_predecessors = {node_id: 0 for node_id in range(self.dag.node_num)}
        self.successors = {node_id: set() for node_id in range(self.dag.node_num)}

        for predecessor, successor in self.dag.edges:
            self.remaining_predecessors[successor] += 1
            self.successors[predecessor].add(successor)

        self.ready_node_ids = {
            node_id
            for node_id, predecessor_count in self.remaining_predecessors.items()
            if predecessor_count == 0
        }

    def mark_finished(self, node_id: int, finish_time: float) -> None:
        """记录一个就绪子任务完成，并释放其已满足全部依赖的后继。"""
        if self.failed_at is not None:
            raise RuntimeError("失败 DAG 不能继续执行")
        if node_id not in self.ready_node_ids:
            raise ValueError(f"子任务 {node_id} 当前并非就绪状态，不能完成。")

        self.ready_node_ids.remove(node_id)
        self.finished_at[node_id] = finish_time

        for successor in self.successors[node_id]:
            self.remaining_predecessors[successor] -= 1
            if self.remaining_predecessors[successor] == 0:
                self.ready_node_ids.add(successor)

    def mark_failed(self, at: float) -> None:
        """截止时刻终止未完成 DAG，保留已完成节点的记录。"""
        if not self.is_finished and self.failed_at is None:
            self.failed_at = float(at)
            self.ready_node_ids.clear()

    @property
    def is_finished(self) -> bool:
        """所有子任务是否都已完成。"""
        return len(self.finished_at) == self.dag.node_num

    @property
    def completion_time(self) -> float | None:
        """DAG 完成时刻；尚未完成时返回 ``None``。"""
        if not self.is_finished:
            return None
        return max(self.finished_at.values(), default=0.0)
