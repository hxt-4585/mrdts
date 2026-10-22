"""按智能体可见范围构建当前状态观测。"""

import numpy as np

from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV


class MasterObservation:
    """构建单架 Master 可见的 Member 状态矩阵。"""

    def __init__(self, region: Region, masters: MasterUAV, members: MemberUAV):
        self.region = region
        self.masters = masters
        self.members = members

    def build(self, master_id: int) -> np.ndarray:
        """返回按 Member ID 排列的 (M, 3) 矩阵；不可见行置零。"""
        region_id = int(self.masters.region_ids[master_id])
        visible_region_ids = (region_id, *self.masters.neighbor_region_ids[region_id])
        member_count = self.members.member_uav_count
        visible = np.isin(self.members.region_ids[:member_count], visible_region_ids)

        observation = np.zeros((member_count, 3), dtype=float)
        np.divide(
            self.members.positions[:member_count, :2],
            self.region.config.side_length,
            out=observation[:, :2],
            where=visible[:, None],
        )
        np.divide(
            self.members.energy_virtual_queue_j[:member_count],
            self.members.config.member_slot_energy,
            out=observation[:, 2],
            where=visible,
        )
        return observation
