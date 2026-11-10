"""子任务的传输、数据到达和计算生命周期。"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from env.types import EntityRef, TaskKey


class TaskStatus(str, Enum):
    UNSCHEDULED = "unscheduled"
    WAITING_DATA = "waiting_data"
    QUEUED_COMPUTE = "queued_compute"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class TransferRecord:
    transfer_ids: tuple[str, ...]
    start_at: float | None = None
    finish_at: float | None = None
    tx_energy_j: float = 0.0  # 路由中 Member UAV 的发送能耗，与窗口统计口径一致。


@dataclass
class TaskRuntime:
    key: TaskKey
    cpu_cycles: float
    input_bits: float
    predecessors: set[TaskKey]
    execution_node: EntityRef | None = None
    decision_at: float | None = None
    status: TaskStatus = field(default=TaskStatus.UNSCHEDULED, init=False)
    input_record: TransferRecord | None = None
    predecessor_records: dict[TaskKey, TransferRecord] = field(default_factory=dict)
    input_arrival_at: float | None = None
    predecessor_arrival_at: dict[TaskKey, float] = field(default_factory=dict)
    data_ready_at: float | None = None
    compute_queue_enter_at: float | None = None
    compute_start_at: float | None = None
    compute_finish_at: float | None = None
    execution_core_id: int | None = None
    compute_energy_j: float = 0.0
    member_energy_j: np.ndarray | None = None  # 当前任务的逐 Member 计算与发送能耗。
    energy_virtual_queue_j: np.ndarray | None = None  # 完成时快照，尚未扣时隙预算。
    failed_at: float | None = None
    unfinished_dags_at_decision: int | None = None
    unfinished_region_dags_at_decision: int | None = None  # 决策时 owner 所在区域的未完成 DAG 总数。
    managed_dag_count: int | None = None

    def mark_input_arrived(self, at: float) -> bool:
        self.input_arrival_at = float(at)
        return self._update_data_ready()

    def mark_predecessor_arrived(self, predecessor: TaskKey, at: float) -> bool:
        self.predecessor_arrival_at[predecessor] = float(at)
        return self._update_data_ready()

    def mark_compute_started(self, core_id: int, at: float) -> None:
        self.status = TaskStatus.RUNNING
        self.execution_core_id = core_id
        self.compute_start_at = float(at)

    def mark_compute_finished(self, at: float) -> None:
        self.status = TaskStatus.FINISHED
        self.compute_finish_at = float(at)

    def _update_data_ready(self) -> bool:
        if self.data_ready_at is not None:
            return False
        if self.input_arrival_at is None or len(self.predecessor_arrival_at) != len(self.predecessors):
            return False
        self.data_ready_at = max([self.input_arrival_at, *self.predecessor_arrival_at.values()])
        self.compute_queue_enter_at = self.data_ready_at
        self.status = TaskStatus.QUEUED_COMPUTE
        return True
