"""服务器级 FIFO 多核计算调度状态。"""

from collections import deque
from dataclasses import dataclass, field

from env.types import EntityRef, TaskKey


@dataclass(frozen=True)
class ComputeRecord:
    task_key: TaskKey
    core_id: int
    start_at: float
    finish_at: float
    energy_j: float


@dataclass(frozen=True)
class _QueuedCompute:
    task_key: TaskKey
    cpu_cycles: float
    ready_at: float
    priority_seq: int


@dataclass
class ServerState:
    node: EntityRef
    core_frequencies: tuple[float, ...]
    capacitance_factor: float
    queue: deque[_QueuedCompute] = field(default_factory=deque)
    running: dict[int, ComputeRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.core_frequencies or any(value <= 0.0 for value in self.core_frequencies):
            raise ValueError("服务器至少需要一个正频率核心")
        if self.capacitance_factor < 0.0:
            raise ValueError("capacitance_factor 必须非负")

    @property
    def queued_task_keys(self) -> tuple[TaskKey, ...]:
        return tuple(item.task_key for item in self.queue)

    def enqueue(self, task_key: TaskKey, cpu_cycles: float, ready_at: float, priority_seq: int) -> None:
        if cpu_cycles <= 0.0:
            raise ValueError("cpu_cycles 必须为正")
        if self.queue and (ready_at, priority_seq, task_key) < (
            self.queue[-1].ready_at,
            self.queue[-1].priority_seq,
            self.queue[-1].task_key,
        ):
            raise ValueError("服务器 FIFO 入队顺序不可倒退")
        self.queue.append(_QueuedCompute(task_key, float(cpu_cycles), float(ready_at), priority_seq))

    def dispatch(self, now: float) -> tuple[ComputeRecord, ...]:
        idle = [core_id for core_id in range(len(self.core_frequencies)) if core_id not in self.running]
        started: list[ComputeRecord] = []
        while idle and self.queue:
            item = self.queue[0]
            if item.ready_at > now:
                break
            self.queue.popleft()
            core_id = idle.pop(0)
            frequency = self.core_frequencies[core_id]
            finish_at = now + item.cpu_cycles / frequency
            energy_j = self.capacitance_factor * item.cpu_cycles * frequency**2
            record = ComputeRecord(item.task_key, core_id, now, finish_at, energy_j)
            self.running[core_id] = record
            started.append(record)
        return tuple(started)

    def finish(self, core_id: int, at: float) -> ComputeRecord:
        record = self.running.get(core_id)
        if record is None:
            raise KeyError(f"核心 {core_id} 当前没有运行任务")
        if abs(record.finish_at - at) > 1e-12:
            raise ValueError("核心只能在任务预定结束时刻释放")
        del self.running[core_id]
        return record
