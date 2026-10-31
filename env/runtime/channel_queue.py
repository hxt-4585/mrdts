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
    """单次传输持续占用 FIFO 队首，资源份额可变；finish_at 仅记录实际完成。"""

    transfer_id: str
    source_ready_at: float | None
    payload_bits: float
    channel_gain: float = 0.0
    remaining_bits: float = field(init=False)
    bandwidth_hz: float = field(default=0.0, init=False)
    power_w: float = field(default=0.0, init=False)
    rate_bps: float = field(default=0.0, init=False)
    predicted_finish_at: float | None = field(default=None, init=False)
    tx_energy_j: float = field(default=0.0, init=False)  # 本跳实际发送能耗，含所有设备类型。
    status: TransferStatus = field(default=TransferStatus.QUEUED, init=False)
    start_at: float | None = field(default=None, init=False)
    finish_at: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.source_ready_at is not None:
            self.source_ready_at = float(self.source_ready_at)
        self.payload_bits = float(self.payload_bits)
        self.remaining_bits = self.payload_bits


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
        self.active = head
        return head

    def finish_active(self, at: float) -> TransferJob:
        at = float(at)
        finished = self.active
        finished.status = TransferStatus.FINISHED
        finished.finish_at = at
        finished.remaining_bits = 0.0
        finished.bandwidth_hz = finished.power_w = finished.rate_bps = 0.0
        self.active = None
        self.next_free_at = at
        return finished
