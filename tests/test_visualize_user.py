"""用户分布可视化脚本的测试。"""

import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.region import Region
from env.user import User
from scripts.visualize_user import draw_user_distribution


class TestUserDistributionDrawing(unittest.TestCase):
    def test_draw_user_distribution_plots_all_users(self):
        """散点图应包含所有用户，并以米作为二维坐标轴单位。"""
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        fig, ax = plt.subplots()
        try:
            scatter = draw_user_distribution(region, users, ax)
            self.assertEqual(len(scatter.get_offsets()), users.num_users)
            self.assertEqual(ax.get_xlabel(), "x (m)")
            self.assertEqual(ax.get_ylabel(), "y (m)")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
