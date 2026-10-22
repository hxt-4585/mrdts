"""服务器级 FIFO 多核计算调度状态。"""

from collections import deque
from dataclasses import dataclass, field
import heapq

from env.types import EntityKind, EntityRef, TaskKey


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

    def compute_wait(self, now: float) -> float:
        """返回此刻就绪的新任务需要等待的秒数，不修改队列或核心状态。

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
        for item in self.queue:
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
            energy_j = (self.capacitance_factor * item.cpu_cycles * frequency**2
                        if self.node.kind is EntityKind.MEMBER_UAV else 0.0)
            record = ComputeRecord(item.task_key, core_id, now, finish_at, energy_j)
            self.running[core_id] = record
            started.append(record)
        return tuple(started)

    def finish(self, core_id: int, at: float) -> ComputeRecord:
        return self.running.pop(core_id)
