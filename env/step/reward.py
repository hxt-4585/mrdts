"""Master 时隙指标和 Member 子任务完成指标。"""

from typing import TYPE_CHECKING

import numpy as np

from env.runtime.slot_result import SlotResult
from env.runtime.task_runtime import TaskRuntime

if TYPE_CHECKING:
    from env.contracts import EndStepResult


class MasterReward:
    """系统级时隙指标：时延奖励、逐 Member 完整能耗和结算后队列。"""

    def __init__(self, scheduling_duration_s: float):
        self.scheduling_duration_s = scheduling_duration_s

    def compute_delay(self, result: SlotResult) -> float:
        """所有 DAG 完成后返回 -sum(C_g - slot_start) / (N * H)。

        调用方保证本窗口的全部 DAG 已完成。每个 DAG 权重相同，统计
        覆盖全部区域和 Member；H 为固定名义调度时隙长度，奖励无量纲。
        """
        return -sum(dag.completion_time - result.slot_start for dag in result.dags) / (
            len(result.dags) * self.scheduling_duration_s
        )

    def compute_energy(self, result: "EndStepResult") -> np.ndarray:
        """返回逐 Member 完整时隙能耗（J），含飞行、悬停、计算和发送。

        直接复用结算快照中的 (M,) 数组，调用方只读使用，不扣除能量预算。
        """
        return result.total_energy_j

    def compute_energy_queue(self, result: "EndStepResult") -> np.ndarray:
        """返回逐 Member 已扣预算并截断至非负的 Q(t+1)（J）。

        直接复用结算快照中的 (M,) 数组，调用方只读使用，不再次更新队列。
        """
        return result.energy_virtual_queue_j


class MemberReward:
    """单个子任务的完成指标；由运行时完成入账后创建。"""

    def __init__(self, task: TaskRuntime, slot_dag_count: int, scheduling_duration_s: float):
        elapsed = task.compute_finish_at - task.decision_at
        self._delay_reward = -task.unfinished_region_dags_at_decision * elapsed / (
            slot_dag_count * scheduling_duration_s
        )
        self._energy_j = task.member_energy_j
        self._energy_queue_j = task.energy_virtual_queue_j

    def compute_delay(self) -> float:
        """返回完成任务的时延奖励 -U_region * Δt / (N * H)。

        U_region 为决策时 owner 所在区域尚未完成的 DAG 总数，含同区域
        其他 Member 管理的 DAG；与实际执行节点的位置无关。N 为本窗口全系统
        登记的 DAG 总数（含无 owner 的 DAG），H 为固定名义调度时隙长度。
        Δt 从提交动作到实际计算完成，包含传输、数据等待、排队和计算。
        """
        return self._delay_reward

    def compute_energy(self) -> np.ndarray:
        """返回当前任务的逐 Member 能耗（J），包含计算及输入/前驱路径发送。

        不包含前驱计算、向后继发送、飞行、悬停及 BS/地面设备能耗。
        复用任务的 (M,) 数组，调用方在任务完成后只读使用。
        """
        return self._energy_j

    def compute_energy_queue(self) -> np.ndarray | None:
        """返回任务完成时各 Member 的队列快照（J），不扣时隙预算。

        同时刻完成事件全部入账后取快照；包含其他任务已入账的能耗。
        独立运行时未接入环境队列时为 None，正常环境始终返回 (M,) 数组。
        """
        return self._energy_queue_j
