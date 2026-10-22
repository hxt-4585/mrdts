"""区域划分模块。

本模块负责将全局仿真区域划分为多个逻辑区域，实现
specs/02_scenario_parameters.md 第 1 节定义的区域划分规则：

- 全局区域由 grid_size × grid_size 个基础网格组成；
- 区域地图采用离散加权质心 Voronoi（CVT）方式生成；
- 每个区域满足：四邻接连通、互不重叠、面积不低于下限，且全覆盖整个区域；
- 区域面积由 ``area_imbalance`` 与 ``min_area_ratio`` 控制；当前默认配置使用
  ``area_imbalance=0.5`` 的不均衡面积场景，只有设为 ``0`` 时才趋近等面积，
  并避免细长通道和随机尖刺。
"""

import numpy as np

from env.settings import RegionConfig


class Region:
    """区域实体：生成区域地图、查询实体区域归属与存取。"""

    def __init__(self, config=None, *, rng):
        self.config = config if config is not None else RegionConfig.default()
        self.rng = rng
        # 区域地图：shape=(grid_size, grid_size)，取值 1..R，0 表示未分配
        self.region_map = None
        # 区域编号 -> 与其共享网格边的区域编号（升序）。
        self.adjacent_region_ids = None

    def _update_adjacent_region_ids(self):
        """根据区域地图中共享边的网格建立双向区域邻接表。"""
        adjacent = {region_id: set() for region_id in range(1, self.config.region_count + 1)}
        region_map = self.region_map
        for first, second in (
            (region_map[:, :-1], region_map[:, 1:]),
            (region_map[:-1, :], region_map[1:, :]),
        ):
            for left, right in zip(first.flat, second.flat):
                if left != right:
                    adjacent[int(left)].add(int(right))
                    adjacent[int(right)].add(int(left))
        self.adjacent_region_ids = {
            region_id: tuple(sorted(neighbors)) for region_id, neighbors in adjacent.items()
        }

    # ------------------------------ 种子选取 ------------------------------ #
    def _select_seeds(self, rng):
        """用最远点采样为每个区域选取一个种子网格，保证种子彼此尽量分散。

        Args:
            rng: numpy 随机数生成器（Generator）。

        Returns:
            shape=(R, 2) 的数组，每行为一个种子的 (i, j) 网格坐标，
            i 为 x 方向索引，j 为 y 方向索引。
        """
        cfg = self.config
        grid_size, R = cfg.grid_size, cfg.region_count

        # 所有网格的 (i, j) 坐标，形状 (N, 2)
        ii, jj = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        coords = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(np.int64)

        # 第一个种子随机选取
        first = int(rng.integers(len(coords)))
        seed_idx = [first]

        # 依次选取离已有种子集合最远的候选作为下一个种子
        while len(seed_idx) < R:
            selected = coords[np.array(seed_idx)]                # (k, 2)
            diff = coords[:, None, :] - selected[None, :, :]     # (N, k, 2)
            min_dist = (diff ** 2).sum(axis=2).min(axis=1)       # 每个候选到最近种子的平方距离
            next_idx = int(np.argmax(min_dist))
            seed_idx.append(next_idx)

        return coords[np.array(seed_idx)]

    # -------------------------- 加权 Voronoi / CVT ------------------------- #
    @staticmethod
    def _cell_coordinates(grid_size):
        """返回按 region_map 行优先顺序排列的所有 (x, y) 网格中心坐标。"""
        y, x = np.indices((grid_size, grid_size), dtype=float)
        return np.column_stack((x.ravel(), y.ravel()))

    def _assign_power_cells(self, seeds, targets, coordinates):
        """以加权平方距离分配网格，并迭代权重以逼近目标面积。

        对第 r 个区域最小化 ``||x - seed_r||² - weight_r``。这就是离散
        power diagram；连续形式的每个单元是凸集，因此不会出现一个区域把
        另一个区域从中间截断的情况。
        """
        R = self.config.region_count
        squared_distances = ((coordinates[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
        weights = np.zeros(R, dtype=float)
        labels = np.zeros(len(coordinates), dtype=np.int32)
        counts = np.zeros(R, dtype=int)

        for _ in range(self.config.capacity_iterations):
            labels = np.argmin(squared_distances - weights, axis=1).astype(np.int32)
            counts = np.bincount(labels, minlength=R)
            errors = targets - counts
            if np.max(np.abs(errors)) <= 1.0:
                break
            # 增大权重会扩大对应 power cell。减去均值不改变相对归属。
            weights += 0.2 * errors
            weights -= weights.mean()

        return labels, counts

    def _compute_target_sizes(self, total):
        """根据面积不均衡程度计算各区域的目标面积。

        目标面积 = 面积下限 + 剩余面积 × 权重，其中权重按几何级数递减，
        使区域 0 的权重最大（对应面积最大的「市中心」区域）。

        - area_imbalance = 0 时，权重均等，各区域目标面积相同；
        - area_imbalance 越接近 1，权重递减越快，面积差异越大。

        Args:
            total: 总网格数。

        Returns:
            shape=(R,) 的浮点数组，各区域目标面积，且满足 sum == total。
        """
        cfg = self.config
        R = cfg.region_count
        base = cfg.min_cells_per_region
        remaining = total - R * base  # 保底之外的剩余可分配面积

        a = cfg.area_imbalance
        if a == 0.0:
            weights = np.full(R, 1.0 / R)
        else:
            q = 1.0 - a  # 几何级数公比
            raw = np.array([q ** r for r in range(R)], dtype=float)
            weights = raw / raw.sum()

        return base + remaining * weights

    def generate(self):
        """生成区域地图（离散加权质心 Voronoi 划分）。

        Lloyd 迭代交替执行两步：根据加权距离为每个网格分区，再将各区域
        种子移到所属网格的质心。与随机前沿扩张不同，区域形状由全局距离场
        决定，因此更紧凑，且不会被其他区域从中间拦截。

        Returns:
            shape=(grid_size, grid_size) 的区域编号矩阵。
        """
        cfg = self.config
        grid_size, R = cfg.grid_size, cfg.region_count
        total = grid_size ** 2

        rng = self.rng
        coordinates = self._cell_coordinates(grid_size)
        seeds = self._select_seeds(rng).astype(float)
        targets = self._compute_target_sizes(total)

        for _ in range(cfg.lloyd_iterations):
            labels, _ = self._assign_power_cells(seeds, targets, coordinates)
            for region_id in range(R):
                seeds[region_id] = coordinates[labels == region_id].mean(axis=0)

        labels, _ = self._assign_power_cells(seeds, targets, coordinates)

        self.region_map = labels.reshape(grid_size, grid_size) + 1
        self._update_adjacent_region_ids()
        return self.region_map

    # ------------------------------ 区域归属查询 ------------------------------ #
    def get_region_id(self, x, y):
        """根据连续坐标 (x, y) 查询实体所属的区域编号（1 开始）。

        先将坐标映射到基础网格，再映射到区域。
        """
        cfg = self.config
        cell = cfg.cell_size
        i = int(np.floor(x / cell))
        j = int(np.floor(y / cell))
        # 边界坐标（如 x == env_size）收敛到最后一个网格
        i = min(max(i, 0), cfg.grid_size - 1)
        j = min(max(j, 0), cfg.grid_size - 1)
        return int(self.region_map[j, i])

    def get_region_sizes(self):
        """返回各区域的网格数（面积），索引 0 对应区域 1。"""
        R = self.config.region_count
        counts = np.bincount(self.region_map.ravel(), minlength=R + 1)
        return counts[1:].astype(int)

    # ------------------------------ 存取 ------------------------------ #

    def save(self, path):
        """将区域地图保存为 .npy 文件。"""
        np.save(path, self.region_map)

    def load(self, path):
        """从 .npy 文件加载区域地图。"""
        region_map = np.load(path)
        self.region_map = region_map
        self._update_adjacent_region_ids()
        return self.region_map
