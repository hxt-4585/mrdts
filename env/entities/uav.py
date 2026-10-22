"""UAV 集合的共享空间状态。"""

import numpy as np

from env.settings import EnvConfig, UAVConfig


class UAV:
    """以数组第一维表示 UAV 编号的同类 UAV 集合。"""

    def __init__(self, positions=None, region_ids=None):
        if positions is None and region_ids is None:
            self.positions = None
            self.region_ids = None
            return
        self.positions = np.asarray(positions, dtype=np.float32)
        self.region_ids = np.asarray(region_ids, dtype=np.int32)

    @property
    def num_uavs(self):
        """返回集合中的 UAV 数量。"""
        return len(self.region_ids)

    def update_positions(self, indices, positions):
        """按 UAV 索引批量更新三维位置。"""
        self.positions[np.asarray(indices)] = np.asarray(positions, dtype=np.float32)

    def update_region_ids(self, indices, region_ids):
        """按 UAV 索引批量更新区域编号。"""
        self.region_ids[np.asarray(indices)] = np.asarray(region_ids, dtype=np.int32)


class MasterUAV(UAV):
    """每个区域一架、位置固定的 Master UAV 集合。"""

    def __init__(self, config=None):
        self.config = config if config is not None else UAVConfig.default()
        self.neighbor_region_ids = None
        super().__init__()

    def generate_from_region(self, region):
        """为每个区域生成一架位于区域内部的固定 Master UAV。"""

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
        self.neighbor_region_ids = dict(region.adjacent_region_ids)
        return self.positions


class MemberUAV(UAV):
    """可在区域间迁移的 Member UAV 集合。"""

    def __init__(self, config=None, env_config=None):
        self.config = config if config is not None else UAVConfig.default()
        self.env_config = env_config if env_config is not None else EnvConfig.default()
        self.region_member_counts = None
        self.core_counts = None
        self.core_frequencies = None
        # 每个计算节点当前时隙的能量虚拟队列 Q_i(t)，最后一行对应固定 BS。
        self.energy_virtual_queue_j = None
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

    def _allocate_region_member_counts(self, region_count):
        """按区域平均分配 Member，余数依次分配给前面的区域。"""
        count, remainder = divmod(self.member_uav_count, region_count)
        counts = np.full(region_count, count, dtype=np.int32)
        counts[:remainder] += 1
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
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        side_length = float(side_length)

        velocities = normalized_actions * self.config.max_horizontal_speed
        member_positions = self.positions[: self.bs_index, :2]
        candidate_positions = member_positions + velocities * self.env_config.flight_duration_s
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
        ordered = available[np.argsort(squared_distances[available], kind="stable")]
        return positions[ordered[:count]]

    def generate_from_region_and_user(self, region, users, masters):
        """按区域平均分配数量，位置只由本区域网格和 Master 决定。"""

        region_count = region.config.region_count
        member_counts = self._allocate_region_member_counts(region_count)
        positions = np.empty((self.member_uav_count + 1, 3), dtype=np.float32)
        region_ids = np.empty(self.member_uav_count + 1, dtype=np.int32)

        start = 0
        for region_index, count in enumerate(member_counts):
            end = start + int(count)
            master_positions = masters.positions[masters.region_ids == region_index + 1, :2]
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
        self.energy_virtual_queue_j = np.zeros(self.num_uavs, dtype=float)
        return self.positions
