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


@dataclass
class ServerState:
    node: EntityRef
    core_frequencies: tuple[float, ...]
    capacitance_factor: float
    queue: deque[_QueuedCompute] = field(default_factory=deque)
    running: dict[int, ComputeRecord] = field(default_factory=dict)

    @property
    def queued_task_keys(self) -> tuple[TaskKey, ...]:
        return tuple(item.task_key for item in self.queue)

    def enqueue(self, task_key: TaskKey, cpu_cycles: float, ready_at: float) -> None:
        self.queue.append(_QueuedCompute(task_key, float(cpu_cycles), float(ready_at)))

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
        return self.running.pop(core_id)
