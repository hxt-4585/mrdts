"""信道模型的单元测试。"""

import math
from dataclasses import replace
import unittest

import numpy as np

from env.communication.channel_model import ChannelModel, LinkType
from env.settings import ChannelConfig


class TestChannelModel(unittest.TestCase):
    """验证各类链路的传播计算与速率。"""

    def setUp(self):
        self.config = ChannelConfig.default()
        self.model = ChannelModel(self.config)

    def test_equal_bandwidth_and_power_scaling_preserves_snr(self):
        full = self.model.calculate_link((0., 0., 50.), (100., 0., 50.),
                                         3., LinkType.AIR_TO_AIR, 12e6)
        third = self.model.calculate_link((0., 0., 50.), (100., 0., 50.),
                                          1., LinkType.AIR_TO_AIR, 4e6)
        self.assertAlmostEqual(third.rate_bps, full.rate_bps / 3.)

    def test_coincident_members_have_finite_rate_and_positive_delay(self):
        position = (0., 0., 50.)
        metrics = self.model.calculate_link(position, position, 1., LinkType.AIR_TO_AIR)
        delay = self.model.transmission_delay_s(500_000., position, position, 1., LinkType.AIR_TO_AIR)
        self.assertEqual(metrics.distance_m, 0.)
        self.assertTrue(math.isfinite(metrics.rate_bps))
        self.assertGreater(delay, 0.)
        self.assertLess(delay, .001)
        shared = self.model.calculate_link(position, position, 1., LinkType.AIR_TO_AIR, 1e6)
        self.assertLess(shared.rate_bps, metrics.rate_bps)

    def test_rate_uses_noise_spectral_density_times_allocated_bandwidth(self):
        model = ChannelModel(replace(self.config, noise_density_w_per_hz=1e-17))
        self.assertAlmostEqual(model.rate_bps(4e-11, 1., 4e6), 4e6)

    def test_all_link_types_use_the_same_total_bandwidth(self):
        """三类链路的独占参考带宽一致，空空链路使用确定 LoS 增益。"""
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

        expected_bandwidth_hz = self.config.total_bandwidth_mhz * 1e6
        self.assertEqual(ground_to_air.bandwidth_hz, expected_bandwidth_hz)
        self.assertEqual(air_to_air.bandwidth_hz, expected_bandwidth_hz)
        self.assertEqual(air_to_ground.bandwidth_hz, expected_bandwidth_hz)
        self.assertGreater(air_to_air.rate_bps, ground_to_air.rate_bps)
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

    def test_calculates_transmission_delay(self):
        """传输时延应由链路速率推导。"""
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

        self.assertAlmostEqual(delay, data_size_bits / metrics.rate_bps)

    def test_allocated_bandwidth_is_used_for_rate(self):
        """运行时传入本次传输实际分配到的带宽。"""
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
