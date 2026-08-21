"""外部配置加载的单元测试。"""

from pathlib import Path
import random
import unittest

import numpy as np

from env.settings import DAGConfig, RegionConfig, UserConfig
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
        self.assertEqual(config.seed, 60)

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
