"""按智能体可见范围构建当前状态观测。"""

from collections.abc import Mapping

import numpy as np

from env.contracts import PlacementDecision
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.runtime.event_runtime import SchedulingRuntime
from env.settings import DAGConfig
from env.types import EntityKind, EntityRef, TaskKey


def _write_member_state(region, members, visible, positions, energy_queues, queue_values):
    """将可见成员的归一化状态写入已置零的输出视图，不包含 BS。"""
    member_count = members.member_uav_count
    np.divide(
        members.positions[:member_count, :2],
        region.config.side_length,
        out=positions,
        where=visible[:, None],
    )
    np.divide(
        queue_values[:member_count],
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
            self.region, self.members, visible, member_state[:, :2], member_state[:, 2],
            self.members.energy_virtual_queue_j,
        )
        return observation


class MemberObservation:
    """构建当前子任务及其所属 Member 的固定长度观测。"""

    def __init__(self, region: Region, members: MemberUAV, dag_config: DAGConfig):
        self.region = region
        self.members = members
        self.dag_config = dag_config
        self._predecessor_data_scale = (dag_config.n - 1) * dag_config.intermediate_data_range[1]
        self._initial_energy_queues = np.empty(members.member_uav_count, dtype=float)

    def capture_initial_energy_queues(self) -> None:
        """飞行能耗入账后、调度开始前保存本窗口固定使用的能量队列。"""
        np.copyto(self._initial_energy_queues, self.members.energy_virtual_queue_j)

    def build(
        self, task_key: TaskKey, runtime: SchedulingRuntime,
        placements: Mapping[TaskKey, PlacementDecision],
    ) -> np.ndarray:
        """返回 (5M+6,) 向量；调用方选择前驱位置均已选定的待调度子任务。

        各段依次为：计算量/C_max、输入 bit/(D_max_Kbit*1000)、
        前驱中间数据量分布 (M+2，按执行节点累加 Kbit 后除以 (n-1)*D_edge_max)、
        当前 Member one-hot (M)、区域位置 (2M，按 ID 存放 x/L、y/L)、
        调度初始区域能量队列 (M，Q_initial/E)、当前用户位置 (2，x/L、y/L)。
        前驱数据量按全局 Member ID、BS、当前任务所属用户排列。
        同一执行节点累加各前驱传给当前子任务的边数据量，无数据时为零。
        Member 位置和能量队列仅显示本区域（含自身），其余置零。
        能量队列在飞行能耗入账后、调度开始前保存，同一窗口内固定。
        """
        tasks = runtime.tasks
        task = tasks[task_key]
        member_id = task_key.owner_member_id
        member_count = self.members.member_uav_count
        ground = EntityRef(EntityKind.GROUND_DEVICE, task_key.user_id)
        observation = np.zeros(5 * member_count + 6, dtype=float)
        observation[0] = task.cpu_cycles / self.dag_config.cpu_range[1]
        observation[1] = task.input_bits / (self.dag_config.input_data_range[1] * 1000.0)

        predecessor_data = observation[2:4 + member_count]
        if task.predecessors:
            dag = runtime.dag_runtimes[member_id, task_key.user_id, task_key.dag_id].dag
            for predecessor in task.predecessors:
                execution_node = placements[predecessor].execution_node
                data_kbit = dag.edge_features[predecessor.node_id, task_key.node_id]
                if execution_node.kind is EntityKind.MEMBER_UAV:
                    predecessor_data[execution_node.index] += data_kbit
                elif execution_node.kind is EntityKind.BS:
                    predecessor_data[member_count] += data_kbit
                elif execution_node == ground:
                    predecessor_data[member_count + 1] += data_kbit
            predecessor_data /= self._predecessor_data_scale
        observation[4 + member_count + member_id] = 1.0

        visible = (
            self.members.region_ids[:member_count] == self.members.region_ids[member_id]
        )
        positions = observation[4 + 2 * member_count:4 + 4 * member_count].reshape(member_count, 2)
        energy_queues = observation[4 + 4 * member_count:-2]
        _write_member_state(
            self.region, self.members, visible, positions, energy_queues,
            self._initial_energy_queues,
        )

        np.divide(runtime.entity_positions[ground][:2], self.region.config.side_length, out=observation[-2:])
        return observation
