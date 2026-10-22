"""信道模型的单元测试。"""

import math
import unittest

import numpy as np

from env.communication.channel_model import ChannelModel, LinkType
from env.settings import ChannelConfig


class TestChannelModel(unittest.TestCase):
    """验证各类链路的传播计算与速率。"""

    def setUp(self):
        self.config = ChannelConfig.default()
        self.model = ChannelModel(self.config)

    def test_uses_configured_bandwidth_for_each_link_type(self):
        """三类链路使用各自带宽，空空链路使用确定 LoS 增益。"""
        transmitter = np.array([0.0, 0.0, 0.0])
        receiver = np.array([100.0, 0.0, 50.0])

        ground_to_air = self.model.calculate_link(
            transmitter, receiver, 1.0, LinkType.GROUND_TO_AIR
        )
        air_to_air = self.model.calculate_link(
            receiver, (200.0, 0.0, 100.0), 1.0, LinkType.AIR_TO_AIR
        )
        air_to_ground = self.model.calculate_link(
            receiver, transmitter, 1.0, LinkType.AIR_TO_GROUND
        )

        self.assertEqual(ground_to_air.bandwidth_hz, 4e6)
        self.assertEqual(air_to_air.bandwidth_hz, 6e6)
        self.assertEqual(air_to_ground.bandwidth_hz, 4e6)
        self.assertGreater(air_to_air.rate_bps, ground_to_air.rate_bps * 1.5)
        self.assertAlmostEqual(air_to_ground.rate_bps, ground_to_air.rate_bps)

    def test_calculates_expected_los_weighted_channel_gain(self):
        """平均信道增益应按仰角 LoS 概率混合 LoS/NLoS 增益。"""
        transmitter = np.array([0.0, 0.0, 0.0])
        receiver = np.array([100.0, 0.0, 50.0])

        metrics = self.model.calculate_link(
            transmitter, receiver, 1.0, LinkType.GROUND_TO_AIR
        )

        distance_squared = 100.0**2 + 50.0**2
        elevation = math.degrees(math.atan2(50.0, 100.0))
        los_probability = 1.0 / (
            1.0
            + self.config.los_alpha
            * math.exp(-self.config.los_beta * (elevation - self.config.los_alpha))
        )
        los_gain = self.config.reference_channel_gain / distance_squared
        expected_gain = los_probability * los_gain + (1.0 - los_probability) * (
            self.config.nlos_attenuation_factor * los_gain
        )

        self.assertAlmostEqual(metrics.distance_m, math.sqrt(distance_squared))
        self.assertAlmostEqual(metrics.elevation_angle_deg, elevation)
        self.assertAlmostEqual(metrics.los_probability, los_probability)
        self.assertAlmostEqual(metrics.average_channel_gain, expected_gain)

    def test_air_to_air_is_always_los_at_equal_altitude(self):
        """同高度空空链路仍为确定 LoS，不按零仰角概率衰减。"""
        metrics = self.model.calculate_link(
            (0.0, 0.0, 50.0), (100.0, 0.0, 50.0), 1.0, LinkType.AIR_TO_AIR
        )

        self.assertEqual(metrics.los_probability, 1.0)
        self.assertAlmostEqual(
            metrics.average_channel_gain,
            self.config.reference_channel_gain / 100.0**2,
        )

    def test_calculates_transmission_delay_and_transmit_energy(self):
        """传输时延与发送能耗应由同一链路速率推导。"""
        transmitter = np.array([0.0, 0.0, 0.0])
        receiver = np.array([100.0, 0.0, 50.0])
        transmit_power_w = 2.0
        data_size_bits = 8_000.0
        metrics = self.model.calculate_link(
            transmitter, receiver, transmit_power_w, LinkType.GROUND_TO_AIR
        )

        delay = self.model.transmission_delay_s(
            data_size_bits, transmitter, receiver, transmit_power_w, LinkType.GROUND_TO_AIR
        )
        energy = self.model.transmission_energy_j(
            data_size_bits, transmitter, receiver, transmit_power_w, LinkType.GROUND_TO_AIR
        )

        self.assertAlmostEqual(delay, data_size_bits / metrics.rate_bps)
        self.assertAlmostEqual(energy, transmit_power_w * delay)

    def test_explicit_directional_bandwidth_override_changes_rate(self):
        """运行时可为一条有向实体对使用独立带宽。"""
        transmitter = np.array([0.0, 0.0, 0.0])
        receiver = np.array([100.0, 0.0, 50.0])

        metrics = self.model.calculate_link(
            transmitter,
            receiver,
            1.0,
            LinkType.GROUND_TO_AIR,
            bandwidth_hz=2e6,
        )

        self.assertEqual(metrics.bandwidth_hz, 2e6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
