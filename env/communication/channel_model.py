"""基于位置的简化 LoS/NLoS 通信信道模型。"""

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from env.settings import ChannelConfig


class LinkType(str, Enum):
    """仿真中支持的有向链路类别。"""

    GROUND_TO_AIR = "ground_to_air"
    AIR_TO_AIR = "air_to_air"
    AIR_TO_GROUND = "air_to_ground"


@dataclass(frozen=True)
class LinkMetrics:
    """一次链路计算得到的传播和速率指标。"""

    link_type: LinkType
    distance_m: float
    elevation_angle_deg: float
    los_probability: float
    average_channel_gain: float
    bandwidth_hz: float
    rate_bps: float


class ChannelModel:
    """由收发端三维位置计算链路增益、速率、时延及发送能耗。"""

    def __init__(self, config: ChannelConfig | None = None):
        self.config = config if config is not None else ChannelConfig.default()

    def calculate_link(
        self, transmitter, receiver, transmit_power_w, link_type, bandwidth_hz=None
    ) -> LinkMetrics:
        """计算指定类型链路的 LoS/NLoS 平均增益与香农速率。"""
        tx_position = np.asarray(transmitter, dtype=float)
        rx_position = np.asarray(receiver, dtype=float)
        transmit_power_w = float(transmit_power_w)
        link_type = LinkType(link_type)

        displacement = rx_position - tx_position
        horizontal_distance_m = float(np.linalg.norm(displacement[:2]))
        height_difference_m = abs(float(displacement[2]))
        distance_m = math.hypot(horizontal_distance_m, height_difference_m)

        elevation_angle_deg = math.degrees(
            math.atan2(height_difference_m, horizontal_distance_m)
        )
        los_probability = (
            1.0 if link_type is LinkType.AIR_TO_AIR
            else self._los_probability(elevation_angle_deg)
        )
        # 重合位置仍使用有限的高速率；距离平方下限为 1e-12 m²。
        los_gain = self.config.reference_channel_gain / max(distance_m**2, 1e-12)
        nlos_gain = self.config.nlos_attenuation_factor * los_gain
        average_channel_gain = los_probability * los_gain + (1.0 - los_probability) * nlos_gain
        bandwidth_hz = (
            self.config.total_bandwidth_mhz * 1e6
            if bandwidth_hz is None
            else float(bandwidth_hz)
        )
        rate_bps = self.rate_bps(average_channel_gain, transmit_power_w, bandwidth_hz)

        return LinkMetrics(
            link_type=link_type,
            distance_m=distance_m,
            elevation_angle_deg=elevation_angle_deg,
            los_probability=los_probability,
            average_channel_gain=average_channel_gain,
            bandwidth_hz=bandwidth_hz,
            rate_bps=rate_bps,
        )

    def transmission_delay_s(
        self, data_size_bits, transmitter, receiver, transmit_power_w, link_type, bandwidth_hz=None
    ):
        """返回将 data_size_bits 通过链路发送所需的时间（s）。"""
        metrics = self.calculate_link(
            transmitter, receiver, transmit_power_w, link_type, bandwidth_hz
        )
        return float(data_size_bits) / metrics.rate_bps

    def transmission_energy_j(
        self, data_size_bits, transmitter, receiver, transmit_power_w, link_type, bandwidth_hz=None
    ):
        """返回发送端传输 data_size_bits 所消耗的能量（J）。"""
        transmit_power_w = float(transmit_power_w)
        return transmit_power_w * self.transmission_delay_s(
            data_size_bits, transmitter, receiver, transmit_power_w, link_type, bandwidth_hz
        )

    def _los_probability(self, elevation_angle_deg):
        return 1.0 / (
            1.0
            + self.config.los_alpha
            * math.exp(-self.config.los_beta * (elevation_angle_deg - self.config.los_alpha))
        )

    def rate_bps(self, channel_gain: float, transmit_power_w: float, bandwidth_hz: float) -> float:
        """理想正交信道：r = b log2(1 + p g / (N0 b))；增益为功率增益。"""
        noise_w = self.config.noise_density_w_per_hz * bandwidth_hz
        snr = transmit_power_w * channel_gain / noise_w
        return bandwidth_hz * (math.log1p(snr) / math.log(2.0))
