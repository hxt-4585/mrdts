"""按全部输入就绪时间派发计算，同刻按子任务的提交顺序派发。"""

from dataclasses import dataclass, field
import heapq
from itertools import count

from env.types import EntityKind, EntityRef, TaskKey


@dataclass(frozen=True)
class ComputeRecord:
    task_key: TaskKey
    core_id: int
    start_at: float
    finish_at: float
    energy_j: float


@dataclass(frozen=True, order=True)
class _QueuedCompute:
    ready_at: float
    schedule_index: int
    sequence: int
    task_key: TaskKey = field(compare=False)
    cpu_cycles: float = field(compare=False)


@dataclass
class ServerState:
    node: EntityRef
    core_frequencies: tuple[float, ...]
    capacitance_factor: float
    queue: list[_QueuedCompute] = field(default_factory=list)
    _sequence: count = field(default_factory=count, init=False)
    running: dict[int, ComputeRecord] = field(default_factory=dict)

    def enqueue(self, task_key: TaskKey, cpu_cycles: float, ready_at: float, schedule_index: int = 0) -> None:
        item = _QueuedCompute(float(ready_at), schedule_index, next(self._sequence), task_key, float(cpu_cycles))
        heapq.heappush(self.queue, item)

    def compute_wait(self, now: float) -> float:
        """估计排在当前全部就绪任务之后的任务需要等待的秒数，不修改队列或核心状态。

        调用方已处理 now 时刻的事件；只推算运行中及公共就绪队列的负载，
        不计等待数据的任务和未来到达。核心同时空闲时沿用编号顺序。
        """
        if not self.queue:
            if len(self.running) < len(self.core_frequencies):
                return 0.0
            return max(0.0, min(record.finish_at for record in self.running.values()) - now)

        available = [
            (self.running[core_id].finish_at if core_id in self.running else now, core_id)
            for core_id in range(len(self.core_frequencies))
        ]
        heapq.heapify(available)
        idle = []
        for item in sorted(self.queue):
            if not idle:
                start_at = available[0][0]
                while available and available[0][0] == start_at:
                    idle.append(heapq.heappop(available)[1])
                idle.reverse()
            # 同一批空闲核心先全部派发，再处理零时长任务的完成事件。
            core_id = idle.pop()
            finish_at = start_at + item.cpu_cycles / self.core_frequencies[core_id]
            heapq.heappush(available, (finish_at, core_id))
        return (start_at if idle else available[0][0]) - now

    def idle_cores(self) -> list[int]:
        return [core_id for core_id in range(len(self.core_frequencies)) if core_id not in self.running]

    def start_next(self, core_id: int, now: float) -> ComputeRecord:
        item = heapq.heappop(self.queue)
        frequency = self.core_frequencies[core_id]
        finish_at = now + item.cpu_cycles / frequency
        energy_j = (self.capacitance_factor * item.cpu_cycles * frequency**2
                    if self.node.kind is EntityKind.MEMBER_UAV else 0.0)
        record = ComputeRecord(item.task_key, core_id, now, finish_at, energy_j)
        self.running[core_id] = record
        return record

    def dispatch(self, now: float) -> tuple[ComputeRecord, ...]:
        started: list[ComputeRecord] = []
        for core_id in self.idle_cores():
            if not self.queue or self.queue[0].ready_at > now:
                break
            started.append(self.start_next(core_id, now))
        return tuple(started)

    def finish(self, core_id: int) -> ComputeRecord:
        return self.running.pop(core_id)
