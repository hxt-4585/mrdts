"""区域划分模块的单元测试。

用法：
    python tests/test_region.py
"""

import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

# 将项目根目录加入 sys.path，便于导入 env 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.randomness import RandomStreams
from env.entities.region import Region
from env.settings import RegionConfig


class TestRegionGeneration(unittest.TestCase):
    """区域划分正确性测试。"""

    def setUp(self):
        self.cfg = RegionConfig.default()

    def _make_manager(self, **kwargs):
        return Region(replace(self.cfg, **kwargs), rng=RandomStreams.from_config().region)

    def test_default_generate_and_validate(self):
        """默认 R=4 生成并通过校验。"""
        mgr = Region(self.cfg, rng=RandomStreams.from_config().region)
        mgr.generate()
        ok, sizes = mgr.validate()
        self.assertTrue(ok)
        self.assertEqual(len(sizes), self.cfg.region_count)

    def test_full_coverage_and_no_overlap(self):
        """地图应全覆盖且每个网格只属于一个区域。"""
        mgr = Region(self.cfg, rng=RandomStreams.from_config().region)
        region_map = mgr.generate()
        # 全覆盖：无 0 值
        self.assertTrue((region_map > 0).all())
        # 取值合法：1..R
        self.assertTrue(
            set(np.unique(region_map).tolist()) <= set(range(1, self.cfg.region_count + 1))
        )

    def test_min_area_ratio(self):
        """每个区域面积不低于下限。"""
        mgr = Region(self.cfg, rng=RandomStreams.from_config().region)
        mgr.generate()
        sizes = mgr.get_region_sizes()
        for r, s in enumerate(sizes):
            self.assertGreaterEqual(s, self.cfg.min_cells_per_region, f"区域 {r + 1} 面积不足")

    def test_region_id_query(self):
        """坐标查询应与区域地图保持一致。"""
        mgr = Region(self.cfg, rng=RandomStreams.from_config().region)
        region_map = mgr.generate()
        cell = self.cfg.cell_size
        # 抽样若干网格中心点查询
        for j in range(0, self.cfg.grid_size, 10):
            for i in range(0, self.cfg.grid_size, 10):
                x = (i + 0.5) * cell
                y = (j + 0.5) * cell
                self.assertEqual(mgr.get_region_id(x, y), int(region_map[j, i]))

    def test_region_num_adjustable(self):
        """区域数量可通过参数调节。"""
        for r in [2, 3, 5]:
            mgr = self._make_manager(region_count=r, min_area_ratio=0.05)
            mgr.generate()
            ok, sizes = mgr.validate()
            self.assertTrue(ok)
            self.assertEqual(len(sizes), r)

    def test_reproducible_with_same_seed(self):
        """相同种子应生成相同地图。"""
        mgr1 = Region(self.cfg, rng=RandomStreams.from_config().region)
        map1 = mgr1.generate()
        mgr2 = Region(self.cfg, rng=RandomStreams.from_config().region)
        map2 = mgr2.generate()
        np.testing.assert_array_equal(map1, map2)

    def test_infeasible_ratio_raises(self):
        """区域数 × 下限 超过总网格数时应抛出 ValueError。"""
        # R=6 且 min_ratio=0.2 -> 6*2000 = 12000 > 10000
        mgr = self._make_manager(region_count=6, min_area_ratio=0.2)
        with self.assertRaises(ValueError):
            mgr.generate()

    def test_region_count_cannot_exceed_grid_cells(self):
        """区域数超过网格数应在生成前被明确拒绝。"""
        mgr = self._make_manager(grid_size=3, region_count=10, min_area_ratio=0.0)
        with self.assertRaises(ValueError):
            mgr.generate()

    def test_area_imbalance_zero_balanced(self):
        """area_imbalance=0 时各区域面积应在合理范围内接近。"""
        mgr = self._make_manager(area_imbalance=0.0)
        mgr.generate()
        sizes = mgr.get_region_sizes()
        self.assertLessEqual(
            int(sizes.max()) - int(sizes.min()),
            int(self.cfg.total_cells * 0.05),
        )

    def test_area_imbalance_creates_size_diff(self):
        """area_imbalance 越大，面积差异应越明显，且满足下限。"""
        mgr = self._make_manager(area_imbalance=0.7)
        mgr.generate()
        sizes = mgr.get_region_sizes()
        # 面积存在明显差异
        self.assertGreater(int(sizes.max()) - int(sizes.min()), 100)
        # 且满足面积下限
        for s in sizes:
            self.assertGreaterEqual(s, mgr.config.min_cells_per_region)
        # 面积最大的应是区域 0（市中心）
        self.assertEqual(int(sizes.argmax()), 0)

    def test_regions_have_compact_boundaries(self):
        """默认区域不应出现随机扩张导致的细长尖刺。"""
        mgr = Region(self.cfg, rng=RandomStreams.from_config().region)
        region_map = mgr.generate()

        for region_id in range(1, self.cfg.region_count + 1):
            cells = region_map == region_id
            area = int(cells.sum())
            padded = np.pad(cells, 1, constant_values=False)
            perimeter = (
                np.count_nonzero(cells & ~padded[1:-1, :-2])
                + np.count_nonzero(cells & ~padded[1:-1, 2:])
                + np.count_nonzero(cells & ~padded[:-2, 1:-1])
                + np.count_nonzero(cells & ~padded[2:, 1:-1])
            )
            # 正方形的该指标为 16；较大的值表示细长边界或尖刺。
            self.assertLess(perimeter ** 2 / area, 25.0)

    def test_area_imbalance_invalid_raises(self):
        """area_imbalance 越界应抛出 ValueError。"""
        for bad in [-0.1, 1.0, 1.5]:
            mgr = self._make_manager(area_imbalance=bad)
            with self.assertRaises(ValueError):
                mgr.generate()

    def test_invalid_cvt_iteration_counts_raise(self):
        """CVT 的两个迭代次数必须为正整数。"""
        for field in ("lloyd_iterations", "capacity_iterations"):
            mgr = self._make_manager(**{field: 0})
            with self.assertRaises(ValueError):
                mgr.generate()

    def test_save_and_load(self):
        """保存后加载的地图应与原地图一致。"""
        mgr = Region(self.cfg, rng=RandomStreams.from_config().region)
        region_map = mgr.generate()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_region.npy")
        try:
            mgr.save(path)
            mgr2 = Region(self.cfg, rng=RandomStreams.from_config().region)
            loaded = mgr2.load(path)
            np.testing.assert_array_equal(region_map, loaded)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_load_rejects_map_with_wrong_shape(self):
        """不能把与当前正方形网格尺寸不一致的地图载入。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_shape.npy"
            np.save(path, np.ones((2, 2), dtype=np.int32))
            with self.assertRaises(ValueError):
                Region(self.cfg, rng=RandomStreams.from_config().region).load(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
