"""同一有向实体对上的严格串行传输队列。"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from env.types import DirectedChannelKey


class TransferStatus(str, Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    FINISHED = "finished"


@dataclass
class TransferJob:
    """等待源数据就绪的单次、不可抢占传输。"""

    transfer_id: str
    source_ready_at: float | None
    duration_s: float
    status: TransferStatus = field(default=TransferStatus.QUEUED, init=False)
    start_at: float | None = field(default=None, init=False)
    finish_at: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.source_ready_at is not None:
            self.source_ready_at = float(self.source_ready_at)
        self.duration_s = float(self.duration_s)


@dataclass
class DirectedChannelState:
    """一个有向信道的 FIFO 队列；仅将源数据已就绪的作业入队。"""

    key: DirectedChannelKey
    queued: deque[TransferJob] = field(default_factory=deque)
    active: TransferJob | None = None
    next_free_at: float = 0.0

    def enqueue(self, job: TransferJob) -> None:
        self.queued.append(job)

    def start_head(self, now: float) -> TransferJob | None:
        now = float(now)
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
        at = float(at)
        finished = self.active
        finished.status = TransferStatus.FINISHED
        self.active = None
        self.next_free_at = at
        return finished
