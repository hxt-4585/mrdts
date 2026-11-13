"""全网连续频谱均分；每个发送端独立均分其总功率预算。"""

from collections import Counter
from typing import Iterator, Mapping, NamedTuple

from env.communication.channel_model import ChannelModel
from env.runtime.channel_queue import TransferJob
from env.types import DirectedChannelKey, EntityRef


class TransferAllocation(NamedTuple):
    job: TransferJob
    bandwidth_hz: float
    power_w: float
    rate_bps: float
    finish_at: float

    def apply(self) -> None:
        self.job.bandwidth_hz = self.bandwidth_hz
        self.job.power_w = self.power_w
        self.job.rate_bps = self.rate_bps
        self.job.predicted_finish_at = self.finish_at


def transfer_allocations(
    active: Mapping[DirectedChannelKey, TransferJob],
    model: ChannelModel,
    transmit_powers: Mapping[EntityRef, float],
    at: float,
) -> Iterator[TransferAllocation]:
    """根据本批候选传输计算资源和完成时间，待同刻事件收齐后生效。"""
    if not active:
        return
    bandwidth_hz = model.config.total_bandwidth_mhz * 1e6 / len(active)
    outgoing = Counter(key.source for key in active)
    for key, job in active.items():
        power_w = transmit_powers[key.source] / outgoing[key.source]
        rate_bps = model.rate_bps(job.channel_gain, power_w, bandwidth_hz)
        yield TransferAllocation(job, bandwidth_hz, power_w, rate_bps, at + job.remaining_bits / rate_bps)


def allocate_transfers(
    active: Mapping[DirectedChannelKey, TransferJob],
    model: ChannelModel,
    transmit_powers: Mapping[EntityRef, float],
    at: float,
) -> None:
    """立即将分配应用到给定的活动传输。"""
    for allocation in transfer_allocations(active, model, transmit_powers, at):
        allocation.apply()
