"""区域可视化脚本的测试。"""

import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.settings import RegionConfig
from scripts.visualize_region import draw_region_map


class TestRegionMapDrawing(unittest.TestCase):
    def test_draw_region_map_adds_grid_line_at_each_cell_boundary(self):
        """每个网格单元的边界都应有一个次刻度位置，供网格线绘制。"""
        config = RegionConfig.default()
        region_map = np.ones((config.grid_size, config.grid_size), dtype=np.int32)
        fig, ax = plt.subplots()
        try:
            draw_region_map(region_map, config, ax)
            expected = np.arange(-0.5, config.grid_size, 1.0)
            np.testing.assert_array_equal(ax.xaxis.get_minorticklocs(), expected)
            np.testing.assert_array_equal(ax.yaxis.get_minorticklocs(), expected)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
