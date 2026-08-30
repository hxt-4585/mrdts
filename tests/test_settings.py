"""外部配置加载的单元测试。"""

from pathlib import Path
import random
import tomllib
import unittest

import numpy as np

from env.settings import ChannelConfig, DAGConfig, RegionConfig, UAVConfig, UserConfig
from env.dag_generator import DAGGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestSettings(unittest.TestCase):
    """TOML 默认配置应加载为类型化设置。"""

    def test_region_config_loads_square_geometry(self):
        config = RegionConfig.from_toml(PROJECT_ROOT / "config" / "region.toml")

        self.assertEqual(config.side_length, 1000.0)
        self.assertEqual(config.grid_size, 100)
        self.assertGreater(config.lloyd_iterations, 0)
        self.assertEqual(config.cell_size, 10.0)
        self.assertEqual(config.region_count, 4)

    def test_default_configs_load_repository_toml_files(self):
        self.assertEqual(RegionConfig.default().grid_size, 100)
        self.assertEqual(DAGConfig.default().n, 10)
        self.assertEqual(UserConfig.default().total_users, 100)

    def test_user_config_loads_population_parameters(self):
        config = UserConfig.from_toml(PROJECT_ROOT / "config" / "user.toml")

        self.assertEqual(config.total_users, 100)
        self.assertEqual(config.min_users_per_region, 1)
        self.assertEqual(config.area_fluctuation, 0.1)
        self.assertEqual(config.center_bias, 0.85)
        self.assertEqual(config.center_spread_ratio, 0.25)
        self.assertEqual(config.transmit_power, 0.1)
        self.assertEqual(config.seed, 60)

    def test_channel_config_loads_link_and_propagation_parameters(self):
        """通信模型配置应保留三类带宽与 LoS/NLoS 参数。"""
        config = ChannelConfig.from_toml(PROJECT_ROOT / "config" / "channel.toml")

        self.assertEqual(config.ground_to_air_bandwidth_mhz, 4.0)
        self.assertEqual(config.air_to_air_bandwidth_mhz, 6.0)
        self.assertEqual(config.air_to_ground_bandwidth_mhz, 4.0)
        self.assertEqual(config.reference_channel_gain, 1e-6)
        self.assertEqual(config.nlos_attenuation_factor, 0.2)
        self.assertEqual(config.noise_power_w, 1e-13)
        self.assertEqual(config.los_alpha, 9.61)
        self.assertEqual(config.los_beta, 0.16)

    def test_uav_config_loads_member_energy_and_communication_parameters(self):
        """推进、计算与通信参数应只作为 Member UAV 的类型化配置加载。"""
        config = UAVConfig.from_toml(PROJECT_ROOT / "config" / "uav.toml")

        self.assertEqual(config.member_propulsion.u1, 85.0)
        self.assertEqual(config.member_propulsion.u2, 0.131)
        self.assertEqual(config.member_propulsion.u3, 0.16)
        self.assertEqual(config.member_propulsion.u4, 0.0115)
        self.assertEqual(config.member_propulsion.u5, 76.0)
        self.assertEqual(config.member_propulsion.tip_speed, 110.0)
        self.assertEqual(config.member_capacitance_factor, 1e-28)
        self.assertEqual(config.member_coverage_radius, 180.0)
        self.assertEqual(config.member_transmit_power, 1.0)

    def test_member_physical_parameters_are_not_nested_in_toml_groups(self):
        """Member 的物理参数应直接位于 [member]，便于统一查阅。"""
        with (PROJECT_ROOT / "config" / "uav.toml").open("rb") as file:
            member = tomllib.load(file)["member"]

        self.assertNotIn("propulsion", member)
        self.assertNotIn("computation", member)
        self.assertNotIn("communication", member)
        self.assertEqual(member["u1"], 85.0)
        self.assertEqual(member["capacitance_factor"], 1e-28)
        self.assertEqual(member["coverage_radius"], 180.0)
        self.assertEqual(member["transmit_power"], 1.0)

    def test_dag_and_region_use_distinct_seeds(self):
        region = RegionConfig.from_toml(PROJECT_ROOT / "config" / "region.toml")
        dag = DAGConfig.from_toml(PROJECT_ROOT / "config" / "dag.toml")

        self.assertIsInstance(region.seed, int)
        self.assertIsInstance(dag.seed, int)
        self.assertNotEqual(region.seed, dag.seed)

    def test_dag_seed_is_independent_of_global_random_state(self):
        config_path = PROJECT_ROOT / "config" / "dag.toml"
        config = DAGConfig.from_toml(config_path)

        np.random.seed(1)
        random.seed(1)
        first = DAGGenerator(config).generate_single_dag()

        np.random.seed(999)
        random.seed(999)
        second = DAGGenerator(config).generate_single_dag()

        self.assertEqual(first.edges, second.edges)
        np.testing.assert_array_equal(first.node_features, second.node_features)
        self.assertEqual(first.edge_features, second.edge_features)


if __name__ == "__main__":
    unittest.main(verbosity=2)
