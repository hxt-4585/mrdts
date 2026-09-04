"""子任务的传输、数据到达和计算生命周期。"""

from dataclasses import dataclass, field
from enum import Enum

from env.types import EntityRef, TaskKey


class TaskStatus(str, Enum):
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
    tx_energy_j: float = 0.0


@dataclass
class TaskRuntime:
    key: TaskKey
    execution_node: EntityRef
    ers_seq: int
    cpu_cycles: float
    input_bits: float
    predecessors: set[TaskKey]
    status: TaskStatus = field(default=TaskStatus.WAITING_DATA, init=False)
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
    failed_at: float | None = None

    def mark_input_arrived(self, at: float) -> bool:
        self.input_arrival_at = float(at)
        return self._update_data_ready()

    def mark_predecessor_arrived(self, predecessor: TaskKey, at: float) -> bool:
        if predecessor not in self.predecessors:
            raise KeyError(f"{predecessor} 不是任务 {self.key} 的前驱")
        self.predecessor_arrival_at[predecessor] = float(at)
        return self._update_data_ready()

    def mark_compute_started(self, core_id: int, at: float, energy_j: float) -> None:
        if self.status is not TaskStatus.QUEUED_COMPUTE:
            raise ValueError("只有已进入计算队列的任务可以启动")
        self.status = TaskStatus.RUNNING
        self.execution_core_id = core_id
        self.compute_start_at = float(at)
        self.compute_energy_j = float(energy_j)

    def mark_compute_finished(self, at: float) -> None:
        if self.status is not TaskStatus.RUNNING:
            raise ValueError("只有运行中的任务可以完成")
        self.status = TaskStatus.FINISHED
        self.compute_finish_at = float(at)

    def _update_data_ready(self) -> bool:
        if self.data_ready_at is not None:
            return False
        if self.input_arrival_at is None or set(self.predecessor_arrival_at) != self.predecessors:
            return False
        self.data_ready_at = max([self.input_arrival_at, *self.predecessor_arrival_at.values()])
        self.compute_queue_enter_at = self.data_ready_at
        self.status = TaskStatus.QUEUED_COMPUTE
        return True
