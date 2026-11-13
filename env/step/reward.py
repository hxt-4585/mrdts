"""Master 时隙指标和 Member 子任务完成指标。"""

from typing import TYPE_CHECKING

import numpy as np

from env.runtime.slot_result import SlotResult

if TYPE_CHECKING:
    from env.contracts import EndStepResult


class MasterReward:
    """系统级时隙指标：时延奖励和逐 Member 完整能耗。"""

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


class MemberReward:
    """窗口结算后，单个子任务的时延奖励与逐 Member 能耗。"""

    def __init__(self, delay_reward: float, energy_j: np.ndarray):
        self._delay_reward = delay_reward
        self._energy_j = energy_j

    def compute_delay(self) -> float:
        """返回 -Δt/H；Δt 是按提交顺序加入此节点后，所属 Member 最晚完工时刻的增量。"""
        return self._delay_reward

    def compute_energy(self) -> np.ndarray:
        """返回当前任务的逐 Member 能耗（J），包含计算及输入/前驱路径发送。

        不包含前驱计算、向后继发送、飞行、悬停及 BS/地面设备能耗。
        复用任务的 (M,) 数组，调用方在任务完成后只读使用。
        """
        return self._energy_j
