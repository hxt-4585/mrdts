"""按智能体可见范围构建当前状态观测。"""

import numpy as np

from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.runtime.event_runtime import SchedulingRuntime
from env.settings import DAGConfig
from env.types import EntityKind, EntityRef, TaskKey


def _write_member_state(region, members, visible, positions, energy_queues):
    """将可见成员的归一化状态写入已置零的输出视图，不包含 BS。"""
    member_count = members.member_uav_count
    np.divide(
        members.positions[:member_count, :2],
        region.config.side_length,
        out=positions,
        where=visible[:, None],
    )
    np.divide(
        members.energy_virtual_queue_j[:member_count],
        members.config.member_slot_energy,
        out=energy_queues,
        where=visible,
    )


class MasterObservation:
    """构建单架 Master 可见的 Member 状态向量。"""

    def __init__(self, region: Region, masters: MasterUAV, members: MemberUAV):
        self.region = region
        self.masters = masters
        self.members = members

    def build(self, master_id: int) -> np.ndarray:
        """返回 (3M,) 向量，按 ID 依次存放 [x/L, y/L, Q/E]。

        仅本区域及相邻区域成员可见，其余成员的三个分量置零。
        """
        region_id = int(self.masters.region_ids[master_id])
        visible_region_ids = (region_id, *self.masters.neighbor_region_ids[region_id])
        member_count = self.members.member_uav_count
        visible = np.isin(self.members.region_ids[:member_count], visible_region_ids)

        observation = np.zeros(3 * member_count, dtype=float)
        member_state = observation.reshape(member_count, 3)
        _write_member_state(
            self.region, self.members, visible, member_state[:, :2], member_state[:, 2]
        )
        return observation


class MemberObservation:
    """构建当前子任务及其所属 Member 的固定长度观测。"""

    def __init__(self, region: Region, members: MemberUAV, dag_config: DAGConfig):
        self.region = region
        self.members = members
        self.dag_config = dag_config
        self._member_nodes = tuple(
            EntityRef(EntityKind.MEMBER_UAV, member_id)
            for member_id in range(members.member_uav_count)
        )

    def build(self, task_key: TaskKey, runtime: SchedulingRuntime) -> np.ndarray:
        """返回 (6M+8,) 向量；调用方选择前驱均已完成的待调度子任务。

        各段依次为：计算量/C_max、输入 bit/(D_max_Kbit*1000)、
        前驱位置 multi-hot (M+2)、当前 Member one-hot (M)、
        区域位置 (2M，按 ID 存放 x/L、y/L)、区域能量队列 (M，Q/E)、
        计算等待时间 (M+2，除以固定调度窗口长度)、当前用户位置 (2，x/L、y/L)。
        前驱位置和等待时间均按全局 Member ID、BS、当前任务所属用户的顺序排列。
        位置、能量队列和 Member 计算等待时间仅显示本区域（含自身），其余置零；
        BS、当前用户的等待时间及用户位置始终可见。等待时间基于 runtime.now 的运行及就绪负载，
        不含传输和新任务自身计算时间，允许超过窗口长度，不截断到 1。
        同一执行位置存在多个前驱也只置 1；用户位置及本地等待项仅对应 task_key.user_id。
        """
        tasks = runtime.tasks
        task = tasks[task_key]
        member_id = task_key.owner_member_id
        member_count = self.members.member_uav_count
        ground = EntityRef(EntityKind.GROUND_DEVICE, task_key.user_id)
        observation = np.zeros(6 * member_count + 8, dtype=float)
        observation[0] = task.cpu_cycles / self.dag_config.cpu_range[1]
        observation[1] = task.input_bits / (self.dag_config.input_data_range[1] * 1000.0)

        predecessor_locations = observation[2:4 + member_count]
        for predecessor in task.predecessors:
            execution_node = tasks[predecessor].execution_node
            if execution_node.kind is EntityKind.MEMBER_UAV:
                predecessor_locations[execution_node.index] = 1.0
            elif execution_node.kind is EntityKind.BS:
                predecessor_locations[member_count] = 1.0
            elif execution_node == ground:
                predecessor_locations[member_count + 1] = 1.0
        observation[4 + member_count + member_id] = 1.0

        visible = (
            self.members.region_ids[:member_count] == self.members.region_ids[member_id]
        )
        positions = observation[4 + 2 * member_count:4 + 4 * member_count].reshape(member_count, 2)
        energy_queues = observation[4 + 4 * member_count:4 + 5 * member_count]
        _write_member_state(self.region, self.members, visible, positions, energy_queues)

        compute_wait = observation[4 + 5 * member_count:-2]
        duration = runtime.deadline - runtime.slot_start
        for member_id, node in enumerate(self._member_nodes):
            if visible[member_id]:
                compute_wait[member_id] = runtime.servers[node].compute_wait(runtime.now) / duration
        compute_wait[member_count] = runtime.servers[runtime.global_bs].compute_wait(runtime.now) / duration
        compute_wait[member_count + 1] = runtime.servers[ground].compute_wait(runtime.now) / duration
        np.divide(runtime.entity_positions[ground][:2], self.region.config.side_length, out=observation[-2:])
        return observation
