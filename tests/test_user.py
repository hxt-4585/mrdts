"""地面用户生成的单元测试。"""

import os
import sys
import unittest
from dataclasses import replace

import numpy as np

# 将项目根目录加入 sys.path，便于导入 env 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.region import Region
from env.settings import RegionConfig, UserConfig
from env.user import User


class TestUser(unittest.TestCase):
    """User 应使用区域地图生成全部用户。"""

    def setUp(self):
        self.region = Region(RegionConfig.default())
        self.region.generate()
        self.config = replace(
            UserConfig.default(),
            total_users=20,
            min_users_per_region=1,
            area_fluctuation=0.1,
            seed=60,
        )

    def test_generate_from_region_assigns_all_users_to_regions(self):
        """用户总数、区域统计和三维地面坐标应相互一致。"""
        users = User(self.config)
        positions = users.generate_from_region(self.region)

        self.assertEqual(positions.shape, (self.config.total_users, 3))
        self.assertTrue((positions[:, 2] == 0.0).all())
        self.assertEqual(users.num_users, self.config.total_users)
        self.assertEqual(len(users.region_user_counts), self.region.config.region_count)
        self.assertEqual(int(users.region_user_counts.sum()), self.config.total_users)
        actual_counts = np.bincount(
            users.region_ids,
            minlength=self.region.config.region_count + 1,
        )[1:]
        np.testing.assert_array_equal(users.region_user_counts, actual_counts)

    def test_generate_from_region_is_reproducible_with_seed(self):
        """相同区域地图和用户种子应生成相同用户集合。"""
        first = User(self.config)
        second = User(self.config)

        first.generate_from_region(self.region)
        second.generate_from_region(self.region)

        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.region_ids, second.region_ids)
        np.testing.assert_array_equal(first.region_user_counts, second.region_user_counts)

    def test_generate_from_region_requires_region_map(self):
        """用户生成必须基于已生成或加载的区域地图。"""
        with self.assertRaises(RuntimeError):
            User(self.config).generate_from_region(Region(RegionConfig.default()))

    def test_num_users_before_generation_names_current_entry_point(self):
        """未生成时的提示应指向当前的用户生成接口。"""
        with self.assertRaisesRegex(RuntimeError, "generate_from_region"):
            _ = User(self.config).num_users


if __name__ == "__main__":
    unittest.main(verbosity=2)
