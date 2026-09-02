"""仿真环境的最小资源与链路能力。"""

import numpy as np

from env.channel_model import ChannelModel
from env.uav import MemberUAV


class Environment:
    """协调现有 UAV 计算资源与信道模型的最小环境内核。

    本类暂不管理时隙、动作、奖励或 DAG 生命周期；这些能力将在环境主线
    完成后逐步加入。时隙内计算资源状态由 ``SchedulingRuntime`` 唯一管理。
    """

    def __init__(
        self,
        members: MemberUAV,
        channel_model: ChannelModel,
        user_transmit_power: float,
        user_core_frequency: float = 1e9,
    ):
        self.members = members
        self.channel_model = channel_model
        self.user_transmit_power = user_transmit_power
        self.user_core_frequency = float(user_core_frequency)
        if not np.isfinite(self.user_core_frequency) or self.user_core_frequency <= 0.0:
            raise ValueError("user_core_frequency 必须为有限正数")
        self.user_member_ids = None

    @property
    def member_transmit_power(self) -> float:
        """返回 Member UAV 的任务转发发射功率（W）。"""
        return self.members.config.member_transmit_power

    @property
    def bs_index(self) -> int:
        """返回统一计算节点矩阵中 BS 的索引。"""
        return self.members.bs_index

    def member_ids_in_region(self, region_id: int) -> np.ndarray:
        """返回指定区域内的 Member UAV 索引，不包含全局 BS。"""
        return np.flatnonzero(self.members.region_ids[: self.bs_index] == region_id)

    def server_position(self, server_id: int) -> np.ndarray:
        """返回指定 Member UAV 或 BS 的三维位置。"""
        return self.members.positions[server_id]

    def core_frequencies(self, server_id: int) -> np.ndarray:
        """返回指定计算节点有效核心的频率。"""
        core_count = self.members.core_counts[server_id]
        return self.members.core_frequencies[server_id, :core_count]

    def transmission_delay_s(
        self,
        data_size_bits: float,
        transmitter: np.ndarray,
        receiver: np.ndarray,
        transmit_power_w: float,
        link_type,
    ) -> float:
        """通过当前信道模型计算单条链路的传输时延。"""
        return self.channel_model.transmission_delay_s(
            data_size_bits, transmitter, receiver, transmit_power_w, link_type
        )

    def create_scheduling_runtime(self, user_positions, user_member_ids):
        """按当前实体快照创建仅供本时隙使用的调度运行时。"""
        from env.channel_queue import EntityKind, EntityRef
        from env.event_runtime import SchedulingRuntime, ServerSpec

        user_positions = np.asarray(user_positions, dtype=float)
        user_member_ids = np.asarray(user_member_ids, dtype=int)
        if user_positions.ndim != 2 or user_positions.shape[1] != 3:
            raise ValueError("user_positions 的 shape 必须为 (N, 3)")
        if user_member_ids.shape != (len(user_positions),):
            raise ValueError("user_member_ids 的 shape 必须为 (N,)")
        positions = {}
        powers = {}
        servers = {}
        for member_id in range(self.bs_index):
            entity = EntityRef(EntityKind.MEMBER_UAV, member_id)
            positions[entity] = self.members.positions[member_id]
            powers[entity] = self.member_transmit_power
            servers[entity] = ServerSpec(
                tuple(self.core_frequencies(member_id)), self.members.config.member_capacitance_factor
            )
        bs = EntityRef(EntityKind.BS, 0)
        positions[bs] = self.members.positions[self.bs_index]
        powers[bs] = self.member_transmit_power
        servers[bs] = ServerSpec(
            tuple(self.core_frequencies(self.bs_index)), self.members.config.member_capacitance_factor
        )
        for user_id, position in enumerate(user_positions):
            if not 0 <= user_member_ids[user_id] < self.bs_index:
                raise ValueError("每个用户必须关联一个有效 Member UAV")
            ground = EntityRef(EntityKind.GROUND_DEVICE, user_id)
            positions[ground] = position
            powers[ground] = self.user_transmit_power
            servers[ground] = ServerSpec((self.user_core_frequency,), 0.0)
        member_regions = {
            EntityRef(EntityKind.MEMBER_UAV, member_id): int(self.members.region_ids[member_id])
            for member_id in range(self.bs_index)
        }
        ground_owner_members = {
            EntityRef(EntityKind.GROUND_DEVICE, user_id): EntityRef(
                EntityKind.MEMBER_UAV, int(member_id)
            )
            for user_id, member_id in enumerate(user_member_ids)
        }
        self.user_member_ids = user_member_ids.copy()
        return SchedulingRuntime(
            self.channel_model,
            positions,
            servers,
            powers,
            member_regions=member_regions,
            ground_owner_members=ground_owner_members,
        )

    def refresh_slot_topology(self, region, user_positions, runtime):
        """在 Member 飞行结束后刷新区域、用户关联和当期运行时快照。

        用户固定在地面；每位用户关联到其当前区域内距离最近的 Member UAV。
        传入的 runtime 只属于当前时隙；正式环境应在下一时隙创建新运行时。
        """
        from env.channel_queue import EntityKind, EntityRef

        user_positions = np.asarray(user_positions, dtype=float)
        if user_positions.ndim != 2 or user_positions.shape[1] != 3:
            raise ValueError("user_positions 的 shape 必须为 (N, 3)")
        if not np.isfinite(user_positions).all():
            raise ValueError("user_positions 必须全部为有限数值")

        member_ids = np.arange(self.bs_index, dtype=np.int32)
        member_region_ids = np.array(
            [region.get_region_id(*position[:2]) for position in self.members.positions[: self.bs_index]],
            dtype=np.int32,
        )
        self.members.update_region_ids(member_ids, member_region_ids)
        user_region_ids = np.array(
            [region.get_region_id(*position[:2]) for position in user_positions], dtype=np.int32
        )
        associations = np.empty(len(user_positions), dtype=np.int32)
        for user_id, (position, region_id) in enumerate(zip(user_positions, user_region_ids)):
            candidates = member_ids[member_region_ids == region_id]
            if len(candidates) == 0:
                raise RuntimeError(f"区域 {region_id} 当前没有可关联的 Member UAV")
            horizontal_distances = np.linalg.norm(
                self.members.positions[candidates, :2] - position[:2], axis=1
            )
            associations[user_id] = candidates[int(np.argmin(horizontal_distances))]

        entity_positions = {
            EntityRef(EntityKind.MEMBER_UAV, int(member_id)): self.members.positions[member_id]
            for member_id in member_ids
        }
        entity_positions[EntityRef(EntityKind.BS, 0)] = self.members.positions[self.bs_index]
        runtime.update_entity_positions(entity_positions)
        runtime.update_topology(
            member_regions={
                EntityRef(EntityKind.MEMBER_UAV, int(member_id)): int(member_region_ids[member_id])
                for member_id in member_ids
            },
            ground_owner_members={
                EntityRef(EntityKind.GROUND_DEVICE, user_id): EntityRef(
                    EntityKind.MEMBER_UAV, int(member_id)
                )
                for user_id, member_id in enumerate(associations)
            },
        )
        self.user_member_ids = associations
        return associations.copy()
