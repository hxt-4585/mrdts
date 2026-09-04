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
    """区域实体：生成区域地图、查询实体区域归属、校验与存取。"""

    def __init__(self, config=None):
        self.config = config if config is not None else RegionConfig.default()
        # 区域地图：shape=(grid_size, grid_size)，取值 1..R，0 表示未分配
        self.region_map = None

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
        min_cells = cfg.min_cells_per_region

        if cfg.side_length <= 0:
            raise ValueError("side_length 必须大于 0")
        if not isinstance(grid_size, int) or grid_size < 1:
            raise ValueError("grid_size 必须是正整数")
        if not isinstance(R, int) or not 1 <= R <= total:
            raise ValueError(f"region_count 必须在 [1, {total}] 内")
        if not 0.0 <= cfg.min_area_ratio <= 1.0:
            raise ValueError("min_area_ratio 应在 [0, 1] 区间内")

        # 参数可行性检查：区域数 × 下限 不能超过总网格数
        if R * min_cells > total:
            raise ValueError(
                f"参数不可行：区域数 {R} × 每区域下限 {min_cells} = {R * min_cells} "
                f"超过总网格数 {total}，请调小 min_area_ratio 或 region_count"
            )

        # 面积不均衡程度校验
        if not 0.0 <= cfg.area_imbalance < 1.0:
            raise ValueError(
                f"area_imbalance 应在 [0, 1) 区间内，当前为 {cfg.area_imbalance}"
            )
        if not isinstance(cfg.lloyd_iterations, int) or cfg.lloyd_iterations < 1:
            raise ValueError("lloyd_iterations 必须是正整数")
        if not isinstance(cfg.capacity_iterations, int) or cfg.capacity_iterations < 1:
            raise ValueError("capacity_iterations 必须是正整数")

        rng = np.random.default_rng(cfg.seed)
        coordinates = self._cell_coordinates(grid_size)
        seeds = self._select_seeds(rng).astype(float)
        targets = self._compute_target_sizes(total)

        for _ in range(cfg.lloyd_iterations):
            labels, counts = self._assign_power_cells(seeds, targets, coordinates)
            if (counts == 0).any():
                raise RuntimeError("加权 Voronoi 迭代产生空区域，请调整区域参数")
            for region_id in range(R):
                seeds[region_id] = coordinates[labels == region_id].mean(axis=0)

        labels, counts = self._assign_power_cells(seeds, targets, coordinates)
        if (counts < min_cells).any():
            raise RuntimeError("无法在给定面积下限下完成加权 Voronoi 区域划分")

        self.region_map = labels.reshape(grid_size, grid_size) + 1
        return self.region_map

    # ------------------------------ 区域归属查询 ------------------------------ #
    def get_region_id(self, x, y):
        """根据连续坐标 (x, y) 查询实体所属的区域编号（1 开始）。

        先将坐标映射到基础网格，再映射到区域。
        """
        if self.region_map is None:
            raise RuntimeError("区域地图尚未生成，请先调用 generate()")
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
        if self.region_map is None:
            raise RuntimeError("区域地图尚未生成，请先调用 generate()")
        R = self.config.region_count
        counts = np.bincount(self.region_map.ravel(), minlength=R + 1)
        return counts[1:].astype(int)

    # ------------------------------ 校验与存取 ------------------------------ #
    def validate(self):
        """校验区域地图是否满足文档约束，返回 (是否合法, 各区域网格数)。

        校验项：全覆盖、互不重叠、区域编号合法、面积不低于下限、四邻接连通。
        """
        if self.region_map is None:
            raise RuntimeError("区域地图尚未生成，请先调用 generate()")
        cfg = self.config
        m = self.region_map
        R = cfg.region_count

        if m.shape != (cfg.grid_size, cfg.grid_size):
            raise ValueError(
                f"区域地图尺寸应为 {(cfg.grid_size, cfg.grid_size)}，实际为 {m.shape}"
            )

        # 1. 全覆盖：不允许存在未分配网格（值为 0）
        if not (m > 0).all():
            raise ValueError("存在未分配的基础网格")

        # 2. 互不重叠：每个网格只有一个取值，由矩阵表示天然满足

        # 3. 区域编号合法：取值应为 1..R，且每个区域至少出现一次
        ids = np.unique(m)
        if not set(ids.tolist()) <= set(range(1, R + 1)):
            raise ValueError(f"区域编号越界: {ids.tolist()}")

        sizes = self.get_region_sizes()
        if (sizes == 0).any():
            raise ValueError("存在未出现的空区域")

        # 4. 面积下限
        for r in range(R):
            if sizes[r] < cfg.min_cells_per_region:
                raise ValueError(
                    f"区域 {r + 1} 面积 {sizes[r]} 低于下限 {cfg.min_cells_per_region}"
                )

        # 5. 连通性：每个区域内部四邻接连通
        if not self._is_each_region_connected():
            raise ValueError("存在不连通的区域")

        return True, sizes

    def _is_each_region_connected(self):
        """检查每个区域内部的网格是否通过四邻接构成单一连通分量（并查集）。"""
        m = self.region_map
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

    def save(self, path):
        """将区域地图保存为 .npy 文件。"""
        if self.region_map is None:
            raise RuntimeError("区域地图尚未生成，请先调用 generate()")
        np.save(path, self.region_map)

    def load(self, path):
        """从 .npy 文件加载区域地图。"""
        region_map = np.load(path)
        expected_shape = (self.config.grid_size, self.config.grid_size)
        if region_map.shape != expected_shape:
            raise ValueError(f"区域地图尺寸应为 {expected_shape}，实际为 {region_map.shape}")
        self.region_map = region_map
        return self.region_map
