"""仿真接受的 DAG 请求与子任务放置决策。"""

from dataclasses import dataclass, field

import numpy as np

from env.types import EntityRef, TaskKey
from env.runtime.slot_result import SlotResult
from env.step.reward import MasterReward, MemberReward
from env.workload.dag_generator import DAG


@dataclass(frozen=True)
class PlacementDecision:
    """智能体对一个子任务给出的执行位置，不预先伪造执行时间。"""

    execution_node: EntityRef


@dataclass(frozen=True)
class DAGRequest:
    """当前时隙登记的 DAG；排序和执行位置由 method 决定。"""

    dag_id: int
    dag: DAG
    owner_member: EntityRef | None  # None 表示无关联 Member，由用户自动本地执行。
    ground_device: EntityRef

    @property
    def key(self) -> tuple[int, int, int]:
        owner_id = -1 if self.owner_member is None else self.owner_member.index
        return owner_id, self.ground_device.index, self.dag_id

    @property
    def task_keys(self) -> tuple[TaskKey, ...]:
        return tuple(TaskKey(*self.key, node) for node in range(self.dag.node_num))


@dataclass(frozen=True)
class BeginStepResult:
    """用于整窗规划的工作负载及逐 Member 越界标记。"""

    requests: tuple[DAGRequest, ...]
    flight_violations: np.ndarray


@dataclass(frozen=True)
class EndStepResult:
    """本步结算快照；各能耗及队列数组均为 (M,)，仅包含 Member。"""

    slot_result: SlotResult
    compute_energy_j: np.ndarray
    tx_energy_j: np.ndarray
    flight_energy_j: np.ndarray
    hover_energy_j: np.ndarray
    total_energy_j: np.ndarray
    energy_virtual_queue_j: np.ndarray
    epoch_done: bool
    scheduling_duration_s: float  # 名义调度时隙长度，不随实际完成时刻延长。
    # 整窗执行完成后按 TaskKey 返回的子任务奖励。
    member_rewards: dict[TaskKey, MemberReward] = field(default_factory=dict)

    @property
    def master_reward(self) -> float:
        """完成全部 DAG 调度后读取：所有 Master 共享的归一化负平均时延。"""
        return MasterReward(self.scheduling_duration_s).compute_delay(self.slot_result)
