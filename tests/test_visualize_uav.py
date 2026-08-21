"""UAV 初始分布可视化脚本的测试。"""

import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.region import Region
from env.uav import MasterUAV, MemberUAV
from env.user import User
from scripts.visualize_uav import draw_uav_distribution


class TestUAVDistributionDrawing(unittest.TestCase):
    def test_draw_uav_distribution_plots_users_masters_and_members(self):
        """同一张图应包含全部用户、Master 与 Member。"""
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()
        members.generate_from_region_and_user(region, users, masters)
        fig, ax = plt.subplots()
        try:
            artists = draw_uav_distribution(region, users, masters, members, ax)
            self.assertEqual(len(artists["users"].get_offsets()), users.num_users)
            self.assertEqual(len(artists["masters"].get_offsets()), masters.num_uavs)
            self.assertEqual(len(artists["members"].get_offsets()), members.num_uavs)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
