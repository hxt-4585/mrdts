"""地面用户生成与状态存取。"""

import numpy as np

from env.settings import UserConfig


class User:
    """基于区域地图生成全部固定地面用户。"""

    def __init__(self, config=None):
        self.config = config if config is not None else UserConfig.default()
        self.positions = None
        self.region_ids = None
        self.region_user_counts = None

    @property
    def num_users(self):
        """返回全部区域中的用户总数。"""
        if self.region_user_counts is None:
            raise RuntimeError("用户尚未生成，请先调用 generate_from_region()")
        return int(self.region_user_counts.sum())

    def _allocate_region_user_counts(self, region_areas, rng):
        """按区域面积及配置浮动分配用户数量。"""
        cfg = self.config
        region_count = len(region_areas)
        if not isinstance(cfg.total_users, int) or cfg.total_users < 0:
            raise ValueError("total_users 必须是非负整数")
        if not isinstance(cfg.min_users_per_region, int) or cfg.min_users_per_region < 0:
            raise ValueError("min_users_per_region 必须是非负整数")
        if not 0.0 <= cfg.area_fluctuation <= 1.0:
            raise ValueError("area_fluctuation 应在 [0, 1] 区间内")
        if cfg.total_users < region_count * cfg.min_users_per_region:
            raise ValueError("total_users 不足以满足每个区域的最小用户数")

        weights = region_areas.astype(float) / region_areas.sum()
        if cfg.area_fluctuation > 0.0:
            weights *= rng.uniform(
                1.0 - cfg.area_fluctuation,
                1.0 + cfg.area_fluctuation,
                size=region_count,
            )
            weights /= weights.sum()

        remaining = cfg.total_users - region_count * cfg.min_users_per_region
        quotas = remaining * weights
        counts = np.floor(quotas).astype(np.int32) + cfg.min_users_per_region
        remainder = cfg.total_users - int(counts.sum())
        for index in np.argsort(-(quotas - np.floor(quotas)))[:remainder]:
            counts[index] += 1
        return counts

    def generate_from_region(self, region):
        """根据已生成的区域地图创建用户坐标、区域编号与区域用户数。"""
        if region.region_map is None:
            raise RuntimeError("区域地图尚未生成，请先调用 Region.generate()")

        region_map = region.region_map
        region_count = region.config.region_count
        rng = np.random.default_rng(self.config.seed)
        region_areas = np.bincount(region_map.ravel(), minlength=region_count + 1)[1:]
        counts = self._allocate_region_user_counts(region_areas, rng)

        positions = np.empty((int(counts.sum()), 3), dtype=np.float32)
        region_ids = np.empty(int(counts.sum()), dtype=np.int32)
        cell_size = region.config.cell_size
        start = 0
        for region_index, count in enumerate(counts):
            end = start + int(count)
            cells = np.argwhere(region_map == region_index + 1)
            selected = cells[rng.integers(len(cells), size=count)]
            offsets = rng.random((count, 2))
            positions[start:end, 0] = (selected[:, 1] + offsets[:, 0]) * cell_size
            positions[start:end, 1] = (selected[:, 0] + offsets[:, 1]) * cell_size
            positions[start:end, 2] = 0.0
            region_ids[start:end] = region_index + 1
            start = end

        self.positions = positions
        self.region_ids = region_ids
        self.region_user_counts = counts
        return self.positions
