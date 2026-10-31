"""全网连续频谱均分；每个发送端独立均分其总功率预算。"""

from collections import Counter
from typing import Mapping

from env.communication.channel_model import ChannelModel
from env.runtime.channel_queue import TransferJob
from env.types import DirectedChannelKey, EntityRef


def allocate_transfers(
    active: Mapping[DirectedChannelKey, TransferJob],
    model: ChannelModel,
    transmit_powers: Mapping[EntityRef, float],
    at: float,
) -> None:
    """只给活动传输分配资源；相邻事件间分配保持不变。"""
    if not active:
        return
    bandwidth_hz = model.config.total_bandwidth_mhz * 1e6 / len(active)
    outgoing = Counter(key.source for key in active)
    for key, job in active.items():
        job.bandwidth_hz = bandwidth_hz
        job.power_w = transmit_powers[key.source] / outgoing[key.source]
        job.rate_bps = model.rate_bps(job.channel_gain, job.power_w, bandwidth_hz)
        job.predicted_finish_at = at + job.remaining_bits / job.rate_bps
