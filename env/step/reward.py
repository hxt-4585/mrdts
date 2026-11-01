"""时延奖励：Master 使用全系统 DAG 均值，Member 使用本地归一化成本。"""

from env.runtime.slot_result import SlotResult
from env.runtime.task_runtime import TaskRuntime


class MasterReward:
    """系统级时隙奖励，所有 Master 共享同一个标量奖励。"""

    def __init__(self, scheduling_duration_s: float):
        self.scheduling_duration_s = scheduling_duration_s

    def compute(self, result: SlotResult) -> float:
        """所有 DAG 完成后返回 -sum(C_g - slot_start) / (N * H)。

        调用方保证本窗口的全部 DAG 已完成。每个 DAG 权重相同，统计
        覆盖全部区域和 Member；H 为固定名义调度时隙长度，奖励无量纲。
        """
        return -sum(dag.completion_time - result.slot_start for dag in result.dags) / (
            len(result.dags) * self.scheduling_duration_s
        )


class MemberReward:
    """子任务级奖励，对应 owner Member 的一次执行位置决策。"""

    def __init__(self, scheduling_duration_s: float):
        self.scheduling_duration_s = scheduling_duration_s

    def compute(self, task: TaskRuntime) -> float:
        """仅对实际计算完成的任务计算 -U_i * Δt / (N_i * H)。

        U_i 为决策观测时 owner Member 尚未完成的 DAG 数；N_i 为本窗口
        登记给该 Member 的 DAG 总数，H 为固定名义调度时隙长度。
        Δt 从提交动作到实际计算完成，包含传输、数据等待、排队和计算。
        """
        elapsed = task.compute_finish_at - task.decision_at
        return -task.unfinished_dags_at_decision * elapsed / (
            task.managed_dag_count * self.scheduling_duration_s
        )
