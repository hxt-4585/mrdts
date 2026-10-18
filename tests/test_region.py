"""区域划分模块的单元测试。

用法：
    python tests/test_region.py
"""

import os
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np

# 将项目根目录加入 sys.path，便于导入 env 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.randomness import RandomStreams
from env.entities.region import Region
from env.settings import RegionConfig


def is_each_region_connected(region_map):
    """检查每个区域内部的网格是否通过四邻接构成单一连通分量（并查集）。"""
    m = region_map
    ny, nx = m.shape
    parent = list(range(nx * ny))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 向右、向下邻居若属于同一区域则合并
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            r = m[j, i]
            if i + 1 < nx and m[j, i + 1] == r:
                union(idx, idx + 1)
            if j + 1 < ny and m[j + 1, i] == r:
                union(idx, idx + nx)

    # 每个区域的所有网格应属于同一根
    region_root = {}
    for j in range(ny):
        for i in range(nx):
            r = int(m[j, i])
            root = find(j * nx + i)
            if r in region_root:
                if region_root[r] != root:
                    return False
            else:
                region_root[r] = root
    return True


class TestRegionGeneration(unittest.TestCase):
    """区域划分正确性测试。"""

    def setUp(self):
        self.cfg = RegionConfig.default()

    def _make_manager(self, **kwargs):
        return Region(replace(self.cfg, **kwargs), rng=RandomStreams().region)

    def test_default_generate_and_connectivity(self):
        """默认 R=4 生成完整且连通的区域。"""
        mgr = Region(self.cfg, rng=RandomStreams().region)
        mgr.generate()
        sizes = mgr.get_region_sizes()
        self.assertTrue(is_each_region_connected(mgr.region_map))
        self.assertEqual(len(sizes), self.cfg.region_count)

    def test_full_coverage_and_no_overlap(self):
        """地图应全覆盖且每个网格只属于一个区域。"""
        mgr = Region(self.cfg, rng=RandomStreams().region)
        region_map = mgr.generate()
        # 全覆盖：无 0 值
        self.assertTrue((region_map > 0).all())
        # 取值合法：1..R
        self.assertTrue(
            set(np.unique(region_map).tolist()) <= set(range(1, self.cfg.region_count + 1))
        )

    def test_min_area_ratio(self):
        """每个区域面积不低于下限。"""
        mgr = Region(self.cfg, rng=RandomStreams().region)
        mgr.generate()
        sizes = mgr.get_region_sizes()
        for r, s in enumerate(sizes):
            self.assertGreaterEqual(s, self.cfg.min_cells_per_region, f"区域 {r + 1} 面积不足")

    def test_region_id_query(self):
        """坐标查询应与区域地图保持一致。"""
        mgr = Region(self.cfg, rng=RandomStreams().region)
        region_map = mgr.generate()
        cell = self.cfg.cell_size
        # 抽样若干网格中心点查询
        for j in range(0, self.cfg.grid_size, 10):
            for i in range(0, self.cfg.grid_size, 10):
                x = (i + 0.5) * cell
                y = (j + 0.5) * cell
                self.assertEqual(mgr.get_region_id(x, y), int(region_map[j, i]))

    def test_adjacent_region_ids_include_shared_edges_but_not_corners(self):
        mgr = self._make_manager(region_count=4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "region.npy")
            np.save(path, np.array([[1, 2], [3, 4]], dtype=np.int32))
            mgr.load(path)

        self.assertEqual(
            mgr.adjacent_region_ids,
            {1: (2, 3), 2: (1, 4), 3: (1, 4), 4: (2, 3)},
        )

    def test_generated_regions_have_adjacency(self):
        mgr = self._make_manager(region_count=2)
        mgr.generate()

        self.assertEqual(mgr.adjacent_region_ids, {1: (2,), 2: (1,)})

    def test_region_num_adjustable(self):
        """区域数量可通过参数调节。"""
        for r in [2, 3, 5]:
            mgr = self._make_manager(region_count=r, min_area_ratio=0.05)
            mgr.generate()
            sizes = mgr.get_region_sizes()
            self.assertTrue(is_each_region_connected(mgr.region_map))
            self.assertTrue((sizes >= mgr.config.min_cells_per_region).all())
            self.assertEqual(len(sizes), r)

    def test_reproducible_with_same_seed(self):
        """相同种子应生成相同地图。"""
        mgr1 = Region(self.cfg, rng=RandomStreams().region)
        map1 = mgr1.generate()
        mgr2 = Region(self.cfg, rng=RandomStreams().region)
        map2 = mgr2.generate()
        np.testing.assert_array_equal(map1, map2)

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
        mgr = Region(self.cfg, rng=RandomStreams().region)
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

    def test_save_and_load(self):
        """保存后加载的地图应与原地图一致。"""
        mgr = Region(self.cfg, rng=RandomStreams().region)
        region_map = mgr.generate()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_region.npy")
        try:
            mgr.save(path)
            mgr2 = Region(self.cfg, rng=RandomStreams().region)
            loaded = mgr2.load(path)
            np.testing.assert_array_equal(region_map, loaded)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
