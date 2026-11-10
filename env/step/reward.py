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

    def __init__(self, task: TaskRuntime, delay_improvement_s: float):
        self._delay_reward = float(delay_improvement_s)
        self._energy_j = task.member_energy_j
        self._energy_queue_j = task.energy_virtual_queue_j

    def compute_delay(self) -> float:
        """返回 owner 平均 DAG 时延改善估计，单位为秒，正值表示改善。

        决策前预测当前任务及剩余任务全本地执行的平均时延；当前任务实际
        完成后保留真实结果，预测剩余任务全本地的平均时延，返回前者减后者。
        两边保留此前已完成 DAG，使用固定 DAG key / 拓扑参考顺序与参考
        回传时延。本地动作收益为零；归一化和 Lyapunov 组合由方法侧决定。
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
