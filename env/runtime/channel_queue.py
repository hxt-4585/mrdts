"""同一有向实体对上的严格串行传输队列。"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math

from env.types import DirectedChannelKey


class TransferStatus(str, Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    FINISHED = "finished"


@dataclass
class TransferJob:
    """等待源数据就绪的单次、不可抢占传输。"""

    transfer_id: str
    priority_seq: int
    source_ready_at: float | None
    duration_s: float
    status: TransferStatus = field(default=TransferStatus.QUEUED, init=False)
    start_at: float | None = field(default=None, init=False)
    finish_at: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.transfer_id, str) or not self.transfer_id:
            raise ValueError("transfer_id 必须为非空字符串")
        if isinstance(self.priority_seq, bool) or not isinstance(self.priority_seq, int) or self.priority_seq < 0:
            raise ValueError("priority_seq 必须为非负整数")
        if self.source_ready_at is not None:
            self.source_ready_at = self.validate_time(self.source_ready_at, "source_ready_at")
        self.duration_s = self.validate_duration(self.duration_s)

    @staticmethod
    def validate_time(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} 必须为有限非负数")
        return value

    @staticmethod
    def validate_duration(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("duration_s 必须为有限正数")
        return value


@dataclass
class DirectedChannelState:
    """一个有向信道的 ERS 队列；不与任何其他信道共享状态。"""

    key: DirectedChannelKey
    queued: deque[TransferJob] = field(default_factory=deque)
    active: TransferJob | None = None
    next_free_at: float = 0.0
    _transfer_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _jobs: dict[str, TransferJob] = field(default_factory=dict, init=False, repr=False)
    _last_enqueued_priority_seq: int = field(default=-1, init=False, repr=False)

    def enqueue(self, job: TransferJob) -> None:
        if job.transfer_id in self._transfer_ids:
            raise ValueError(f"传输作业 {job.transfer_id!r} 已存在于该信道")
        if job.status is not TransferStatus.QUEUED:
            raise ValueError("只能将 QUEUED 状态的传输作业加入信道")
        if job.priority_seq < self._last_enqueued_priority_seq:
            raise ValueError("同一信道的传输必须按非递减 ERS 顺序入队")
        self.queued.append(job)
        self._transfer_ids.add(job.transfer_id)
        self._jobs[job.transfer_id] = job
        self._last_enqueued_priority_seq = job.priority_seq

    def mark_source_ready(self, transfer_id: str, at: float) -> None:
        job = self._jobs.get(transfer_id)
        if job is None:
            raise KeyError(f"信道中不存在传输作业 {transfer_id!r}")
        if job.status is not TransferStatus.QUEUED:
            raise ValueError("只能标记尚未启动传输的源数据就绪时刻")
        at = TransferJob.validate_time(at, "at")
        if job.source_ready_at is not None and job.source_ready_at != at:
            raise ValueError("传输源数据就绪时刻不可被改写")
        job.source_ready_at = at

    def start_head(self, now: float) -> TransferJob | None:
        now = TransferJob.validate_time(now, "now")
        if self.active is not None or not self.queued or now < self.next_free_at:
            return None
        head = self.queued[0]
        if head.source_ready_at is None or head.source_ready_at > now:
            return None
        self.queued.popleft()
        head.status = TransferStatus.ACTIVE
        head.start_at = now
        head.finish_at = now + head.duration_s
        self.active = head
        return head

    def finish_active(self, at: float) -> TransferJob:
        at = TransferJob.validate_time(at, "at")
        if self.active is None:
            raise RuntimeError("当前信道没有活动传输")
        if not math.isclose(at, self.active.finish_at, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("活动传输只能在其预定结束时刻完成")
        finished = self.active
        finished.status = TransferStatus.FINISHED
        self.active = None
        self.next_free_at = at
        return finished
