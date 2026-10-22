"""仿真接受的 DAG 请求与子任务放置决策。"""

from dataclasses import dataclass

import numpy as np

from env.types import EntityRef, TaskKey
from env.runtime.slot_result import SlotResult
from env.workload.dag_generator import DAG


@dataclass(frozen=True)
class PlacementDecision:
    """智能体对一个子任务给出的执行位置，不预先伪造执行时间。"""

    execution_node: EntityRef


@dataclass(frozen=True)
class DAGRequest:
    """当前时隙登记的 DAG；排序和执行位置由 method 决定。"""

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


@dataclass(frozen=True)
class BeginStepResult:
    """飞行后已登记的工作负载、首次可决策成员及逐 Member 越界标记。"""

    requests: tuple[DAGRequest, ...]
    ready_member_ids: tuple[int, ...]
    flight_violations: np.ndarray


@dataclass(frozen=True)
class EndStepResult:
    """本步结算快照；各能耗及队列数组均为 (M,)，仅包含 Member。"""

    slot_result: SlotResult
    compute_energy_j: np.ndarray
    tx_energy_j: np.ndarray
    flight_energy_j: np.ndarray
    hover_energy_j: np.ndarray
    total_energy_j: np.ndarray
    energy_virtual_queue_j: np.ndarray
    epoch_done: bool
