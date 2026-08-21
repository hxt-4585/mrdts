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

    def calculate_link(self, transmitter, receiver, transmit_power_w, link_type) -> LinkMetrics:
        """计算指定类型链路的 LoS/NLoS 平均增益与香农速率。"""
        tx_position = self._validate_position(transmitter, "transmitter")
        rx_position = self._validate_position(receiver, "receiver")
        transmit_power_w = self._validate_positive(transmit_power_w, "transmit_power_w")
        link_type = LinkType(link_type)

        displacement = rx_position - tx_position
        horizontal_distance_m = float(np.linalg.norm(displacement[:2]))
        height_difference_m = abs(float(displacement[2]))
        distance_m = math.hypot(horizontal_distance_m, height_difference_m)
        if distance_m == 0.0:
            raise ValueError("收发端位置不能重合")

        elevation_angle_deg = math.degrees(
            math.atan2(height_difference_m, horizontal_distance_m)
        )
        los_probability = self._los_probability(elevation_angle_deg)
        los_gain = self.config.reference_channel_gain / distance_m**2
        nlos_gain = self.config.nlos_attenuation_factor * los_gain
        average_channel_gain = los_probability * los_gain + (1.0 - los_probability) * nlos_gain
        bandwidth_hz = self._bandwidth_hz(link_type)
        snr = transmit_power_w * average_channel_gain / self.config.noise_power_w
        rate_bps = bandwidth_hz * math.log2(1.0 + snr)

        return LinkMetrics(
            link_type=link_type,
            distance_m=distance_m,
            elevation_angle_deg=elevation_angle_deg,
            los_probability=los_probability,
            average_channel_gain=average_channel_gain,
            bandwidth_hz=bandwidth_hz,
            rate_bps=rate_bps,
        )

    def transmission_delay_s(self, data_size_bits, transmitter, receiver, transmit_power_w, link_type):
        """返回将 data_size_bits 通过链路发送所需的时间（s）。"""
        data_size_bits = self._validate_nonnegative(data_size_bits, "data_size_bits")
        metrics = self.calculate_link(transmitter, receiver, transmit_power_w, link_type)
        return data_size_bits / metrics.rate_bps

    def transmission_energy_j(self, data_size_bits, transmitter, receiver, transmit_power_w, link_type):
        """返回发送端传输 data_size_bits 所消耗的能量（J）。"""
        transmit_power_w = self._validate_positive(transmit_power_w, "transmit_power_w")
        return transmit_power_w * self.transmission_delay_s(
            data_size_bits, transmitter, receiver, transmit_power_w, link_type
        )

    def _los_probability(self, elevation_angle_deg):
        return 1.0 / (
            1.0
            + self.config.los_alpha
            * math.exp(-self.config.los_beta * (elevation_angle_deg - self.config.los_alpha))
        )

    def _bandwidth_hz(self, link_type: LinkType) -> float:
        bandwidth_mhz = {
            LinkType.GROUND_TO_AIR: self.config.ground_to_air_bandwidth_mhz,
            LinkType.AIR_TO_AIR: self.config.air_to_air_bandwidth_mhz,
            LinkType.AIR_TO_GROUND: self.config.air_to_ground_bandwidth_mhz,
        }[link_type]
        return bandwidth_mhz * 1e6

    @staticmethod
    def _validate_position(position, name):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"{name} 必须是包含有限数值的 shape=(3,) 三维位置")
        return position

    @staticmethod
    def _validate_positive(value, name):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须为正数")
        return value

    @staticmethod
    def _validate_nonnegative(value, name):
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} 必须为非负数")
        return value
