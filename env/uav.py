"""UAV 集合的共享空间状态。"""

import numpy as np

from env.settings import UAVConfig


class UAV:
    """以数组第一维表示 UAV 编号的同类 UAV 集合。"""

    def __init__(self, positions=None, region_ids=None):
        if positions is None and region_ids is None:
            self.positions = None
            self.region_ids = None
            return
        if positions is None or region_ids is None:
            raise ValueError("positions 与 region_ids 必须同时提供")
        positions = self._validate_positions(positions)
        region_ids = self._validate_region_ids(region_ids)
        if len(positions) != len(region_ids):
            raise ValueError("位置数组和区域编号数组的 UAV 数量必须一致")
        self.positions = positions
        self.region_ids = region_ids

    @property
    def num_uavs(self):
        """返回集合中的 UAV 数量。"""
        if self.region_ids is None:
            raise RuntimeError("UAV 尚未生成")
        return len(self.region_ids)

    def update_positions(self, indices, positions):
        """按 UAV 索引批量更新三维位置。"""
        indices = self._validate_indices(indices)
        positions = self._validate_positions(positions)
        if len(indices) != len(positions):
            raise ValueError("indices 与 positions 的 UAV 数量必须一致")
        self.positions[indices] = positions

    def update_region_ids(self, indices, region_ids):
        """按 UAV 索引批量更新区域编号。"""
        indices = self._validate_indices(indices)
        region_ids = self._validate_region_ids(region_ids)
        if len(indices) != len(region_ids):
            raise ValueError("indices 与 region_ids 的 UAV 数量必须一致")
        self.region_ids[indices] = region_ids

    @staticmethod
    def _validate_positions(positions):
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("positions 的 shape 必须为 (N, 3)")
        if not np.isfinite(positions).all():
            raise ValueError("positions 必须全部为有限数值")
        return positions

    @staticmethod
    def _validate_region_ids(region_ids):
        region_ids = np.asarray(region_ids, dtype=np.int32)
        if region_ids.ndim != 1:
            raise ValueError("region_ids 的 shape 必须为 (N,)")
        if (region_ids < 0).any():
            raise ValueError("region_ids 必须为非负整数")
        return region_ids

    def _validate_indices(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("indices 必须是一维整数数组")
        if ((indices < 0) | (indices >= self.num_uavs)).any():
            raise IndexError("indices 包含超出 UAV 编号范围的索引")
        return indices


class MasterUAV(UAV):
    """每个区域一架、位置固定的 Master UAV 集合。"""

    def __init__(self, config=None):
        self.config = config if config is not None else UAVConfig.default()
        super().__init__()

    def generate_from_region(self, region):
        """为每个区域生成一架位于区域内部的固定 Master UAV。"""
        if region.region_map is None:
            raise RuntimeError("区域地图尚未生成，请先调用 Region.generate()")
        if self.config.master_uav_count != region.config.region_count:
            raise ValueError("Master UAV 数量必须与区域数量一致")

        positions = np.empty((self.config.master_uav_count, 3), dtype=np.float32)
        for region_index in range(self.config.master_uav_count):
            cells = np.argwhere(region.region_map == region_index + 1)
            centroid = cells.mean(axis=0)
            center_cell = cells[np.argmin(((cells - centroid) ** 2).sum(axis=1))]
            positions[region_index] = (
                (center_cell[1] + 0.5) * region.config.cell_size,
                (center_cell[0] + 0.5) * region.config.cell_size,
                self.config.master_altitude,
            )

        region_ids = np.arange(1, self.config.master_uav_count + 1, dtype=np.int32)
        super().__init__(positions, region_ids)
        return self.positions


class MemberUAV(UAV):
    """可在区域间迁移的 Member UAV 集合。"""

    def __init__(self, config=None):
        self.config = config if config is not None else UAVConfig.default()
        self.region_member_counts = None
        self.core_counts = None
        self.core_frequencies = None
        super().__init__()

    @property
    def member_uav_count(self):
        """返回可移动 Member UAV 的数量，不包含最后一行的 BS。"""
        return self.config.member_uav_count

    @property
    def bs_index(self):
        """返回统一计算节点矩阵中固定 BS 的最后一行索引。"""
        return self.member_uav_count

    def _initialize_compute_resources(self):
        """初始化 Member 与最后一行 BS 的计算核心矩阵。"""
        self.core_counts = np.full(self.num_uavs, self.config.member_core_count, dtype=np.int32)
        self.core_counts[self.bs_index] = self.config.bs_core_count
        max_core_count = int(self.core_counts.max())
        self.core_frequencies = np.zeros((self.num_uavs, max_core_count), dtype=float)
        self.core_frequencies[: self.bs_index, : self.config.member_core_count] = (
            self.config.member_core_frequency
        )
        self.core_frequencies[self.bs_index, : self.config.bs_core_count] = (
            self.config.bs_core_frequency
        )

    def _allocate_region_member_counts(self, region_user_counts):
        """按区域用户数分配 Member，并保证区域最低配额。"""
        region_count = len(region_user_counts)
        if not isinstance(self.config.member_uav_count, int):
            raise ValueError("member_uav_count 必须是整数")
        if not isinstance(self.config.min_members_per_region, int):
            raise ValueError("min_members_per_region 必须是整数")
        if self.config.min_members_per_region < 0:
            raise ValueError("min_members_per_region 必须是非负整数")
        minimum = region_count * self.config.min_members_per_region
        if self.config.member_uav_count < minimum:
            raise ValueError("Member UAV 总数不足以满足每个区域的最低配额")

        remaining = self.config.member_uav_count - minimum
        weights = region_user_counts.astype(float)
        if weights.sum() == 0:
            weights.fill(1.0 / region_count)
        else:
            weights /= weights.sum()
        quotas = remaining * weights
        counts = np.floor(quotas).astype(np.int32) + self.config.min_members_per_region
        remainder = self.config.member_uav_count - int(counts.sum())
        for index in np.argsort(-(quotas - np.floor(quotas)))[:remainder]:
            counts[index] += 1
        return counts

    def apply_flight_actions(self, normalized_actions, side_length):
        """应用合法的水平飞行动作，并返回每架 Member 的越界标记。

        Args:
            normalized_actions: shape=(M, 2) 的归一化动作矩阵，元素位于 [-1, 1]。
            side_length: 正方形部署区域的边长；合法水平坐标位于闭区间
                ``[0, side_length]``。

        Returns:
            shape=(M,) 的布尔数组。越界动作对应 ``True``，且该 Member
            保持原位置；合法动作对应 ``False`` 并更新位置。
        """
        if self.positions is None:
            raise RuntimeError("Member UAV 尚未生成")
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        if normalized_actions.shape != (self.member_uav_count, 2):
            raise ValueError("normalized_actions 的 shape 必须为 (M, 2)")
        if not np.isfinite(normalized_actions).all():
            raise ValueError("normalized_actions 必须全部为有限数值")
        if ((normalized_actions < -1.0) | (normalized_actions > 1.0)).any():
            raise ValueError("normalized_actions 的元素必须位于 [-1, 1]")
        if self.config.flight_duration < 0.0:
            raise ValueError("flight_duration 必须为非负数")
        if self.config.max_horizontal_speed < 0.0:
            raise ValueError("max_horizontal_speed 必须为非负数")
        side_length = float(side_length)
        if not np.isfinite(side_length) or side_length <= 0.0:
            raise ValueError("side_length 必须为有限正数")

        velocities = normalized_actions * self.config.max_horizontal_speed
        member_positions = self.positions[: self.bs_index, :2]
        candidate_positions = member_positions + velocities * self.config.flight_duration
        in_bounds = (
            (candidate_positions >= 0.0) & (candidate_positions <= side_length)
        ).all(axis=1)
        member_positions[in_bounds] = candidate_positions[in_bounds]
        return ~in_bounds

    @staticmethod
    def _select_positions_near_master(region, region_id, master_position, count):
        """选择 Master 周围本区域的不同网格中心，水平间距至少一格。

        距 Master 至少保留一格距离，避免相同高度时与 Master 重合。
        相同距离按网格顺序稳定选择；空间不足时拒绝复用已有位置。
        """
        cells = np.argwhere(region.region_map == region_id)
        positions = (cells[:, ::-1] + 0.5) * region.config.cell_size
        squared_distances = ((positions - master_position) ** 2).sum(axis=1)
        available = np.flatnonzero(squared_distances >= region.config.cell_size**2)
        if len(available) < count:
            raise ValueError(f"区域 {region_id} 的可用网格不足，无法保持 Member 初始间距")
        ordered = available[np.argsort(squared_distances[available], kind="stable")]
        return positions[ordered[:count]]

    def generate_from_region_and_user(self, region, users, masters):
        """按区域用户数分配数量，位置只由本区域网格和 Master 决定。"""
        if region.region_map is None:
            raise RuntimeError("区域地图尚未生成，请先调用 Region.generate()")
        if users.region_ids is None:
            raise RuntimeError("用户尚未生成，请先调用 User.generate_from_region()")
        if masters.positions is None or masters.region_ids is None:
            raise RuntimeError("Master UAV 尚未生成，请先调用 MasterUAV.generate_from_region()")

        region_count = region.config.region_count
        user_counts = np.bincount(users.region_ids, minlength=region_count + 1)[1:]
        member_counts = self._allocate_region_member_counts(user_counts)
        positions = np.empty((self.member_uav_count + 1, 3), dtype=np.float32)
        region_ids = np.empty(self.member_uav_count + 1, dtype=np.int32)

        start = 0
        for region_index, count in enumerate(member_counts):
            end = start + int(count)
            master_positions = masters.positions[masters.region_ids == region_index + 1, :2]
            if len(master_positions) != 1:
                raise ValueError("每个区域必须恰有一架 Master UAV")
            positions[start:end, :2] = self._select_positions_near_master(
                region, region_index + 1, master_positions[0], int(count)
            )
            positions[start:end, 2] = self.config.member_altitude
            region_ids[start:end] = region_index + 1
            start = end

        positions[self.bs_index] = (
            region.config.side_length / 2,
            region.config.side_length / 2,
            self.config.bs_altitude,
        )
        region_ids[self.bs_index] = 0
        self.region_member_counts = member_counts
        super().__init__(positions, region_ids)
        self._initialize_compute_resources()
        return self.positions
