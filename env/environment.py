"""仿真环境的最小资源与链路能力。"""

import numpy as np

from env.channel_model import ChannelModel
from env.uav import MemberUAV


class Environment:
    """协调现有 UAV 计算资源与信道模型的最小环境内核。

    本类暂不管理时隙、动作、奖励或 DAG 生命周期；这些能力将在环境主线
    完成后逐步加入。计算资源状态仍归 ``MemberUAV`` 所有。
    """

    def __init__(
        self,
        members: MemberUAV,
        channel_model: ChannelModel,
        user_transmit_power: float,
    ):
        self.members = members
        self.channel_model = channel_model
        self.user_transmit_power = user_transmit_power

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

    def estimate_finish_time(
        self, server_id: int, cpu_cycles: float, data_ready_time: float
    ) -> tuple[float, float]:
        """预估最早可用核心上的开始与完成时刻，不修改资源状态。"""
        core_count = self.members.core_counts[server_id]
        core_available_at = self.members.core_available_at[server_id, :core_count]
        core_id = int(np.argmin(core_available_at))
        start_time = max(data_ready_time, core_available_at[core_id])
        finish_time = start_time + cpu_cycles / self.members.core_frequencies[server_id, core_id]
        return start_time, finish_time

    def reserve_computation(
        self, server_id: int, cpu_cycles: float, data_ready_time: float
    ) -> float:
        """在指定计算节点占用最早可用核心，并返回任务完成时刻。"""
        return self.members.schedule_computation(server_id, cpu_cycles, data_ready_time)

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
