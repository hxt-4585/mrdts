"""仿真接受的 DAG 请求与子任务放置决策。"""

from dataclasses import dataclass

from env.types import EntityRef, TaskKey
from env.workload.dag_generator import DAG


@dataclass(frozen=True)
class PlacementDecision:
    """智能体对一个子任务给出的执行位置，不预先伪造执行时间。"""

    execution_node: EntityRef
    ers_seq: int


@dataclass(frozen=True)
class DAGRequest:
    """当前时隙待排序的 DAG；执行位置由后续卸载策略选择。"""

    dag_id: int
    dag: DAG
    owner_member: EntityRef
    ground_device: EntityRef

    @property
    def key(self) -> tuple[int, int, int]:
        return self.owner_member.index, self.ground_device.index, self.dag_id

    @property
    def task_keys(self) -> tuple[TaskKey, ...]:
        return tuple(TaskKey(*self.key, node) for node in range(self.dag.node_num))
