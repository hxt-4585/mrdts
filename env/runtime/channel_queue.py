"""同一有向实体对上的严格串行传输队列。"""

from dataclasses import dataclass, field
from enum import Enum
import heapq
from itertools import count

from env.types import DirectedChannelKey


class TransferStatus(str, Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    FINISHED = "finished"


@dataclass
class TransferJob:
    """传输启动后持续占用信道，资源份额可变；finish_at 仅记录实际完成。"""

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
    """按本跳源数据就绪时间派发传输，同刻按目标子任务的提交顺序派发。"""

    key: DirectedChannelKey
    queued: list[tuple[float, int, int, TransferJob]] = field(default_factory=list)
    _sequence: count = field(default_factory=count, init=False)
    active: TransferJob | None = None
    next_free_at: float = 0.0

    def enqueue(self, job: TransferJob, schedule_index: int = 0) -> None:
        heapq.heappush(self.queued, (job.source_ready_at, schedule_index, next(self._sequence), job))

    def ready_head(self, now: float) -> TransferJob | None:
        if self.active is not None or not self.queued or now < self.next_free_at:
            return None
        ready_at, _, _, head = self.queued[0]
        if ready_at > now:
            return None
        return head

    def start_head(self, now: float) -> TransferJob | None:
        head = self.ready_head(now)
        if head is None:
            return None
        heapq.heappop(self.queued)
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
