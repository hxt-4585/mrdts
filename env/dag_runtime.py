"""单个活动 DAG 的运行时依赖状态。

本模块不负责卸载决策、传输、计算资源或 UAV 状态；它只追踪哪些子任务
已经满足前驱依赖、何时完成，以及整个 DAG 何时完成。
"""

from dataclasses import dataclass, field

from env.dag_generator import DAG


@dataclass
class DAGRuntime:
    """一个 DAG 在仿真中的轻量运行时状态。"""

    dag: DAG
    remaining_predecessors: dict[int, int] = field(init=False)
    successors: dict[int, set[int]] = field(init=False)
    finished_at: dict[int, float] = field(default_factory=dict, init=False)
    ready_node_ids: set[int] = field(init=False)

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
        if node_id not in self.ready_node_ids:
            raise ValueError(f"子任务 {node_id} 当前并非就绪状态，不能完成。")

        self.ready_node_ids.remove(node_id)
        self.finished_at[node_id] = finish_time

        for successor in self.successors[node_id]:
            self.remaining_predecessors[successor] -= 1
            if self.remaining_predecessors[successor] == 0:
                self.ready_node_ids.add(successor)

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
