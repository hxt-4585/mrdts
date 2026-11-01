"""分层奖励框架；公式尚未实现，也尚未接入环境事件流程。"""

import numpy as np

from env.contracts import EndStepResult
from env.runtime.task_runtime import TaskRuntime


class MasterReward:
    """系统级时隙奖励，所有 Master 共享同一个标量奖励。"""

    def compute(
        self,
        result: EndStepResult,
        *,
        flight_violations: np.ndarray,
    ) -> float:
        """在 end_step 结算后，为整个系统计算一次标量奖励。

        result 为全系统的本时隙结算结果；flight_violations 为 begin_step
        返回的全网 (M,) 越界标记。统计范围覆盖全部区域和 Member，
        不按单个 Master 或其所属区域拆分。
        后续在此确定全系统 DAG 时延、失败、能耗及越界惩罚的组合方式。
        """
        raise NotImplementedError("Master 系统级时隙奖励公式尚未实现")


class MemberReward:
    """子任务级奖励，对应 owner Member 的一次执行位置决策。"""

    def compute(self, task: TaskRuntime) -> float:
        """为一个已提交动作、已完成或截止失败的子任务计算标量奖励。

        调用方以 task.key 将奖励关联到原动作，每个子任务只结算一次；
        不为尚未提交动作的子任务单独生成卸载奖励。
        正常完成时使用 compute_finish_at，截止失败时使用 failed_at，
        相对 decision_at 的间隔包含传输、等待数据、排队和计算时间。
        后续在此确定时延、相关能耗及截止失败惩罚的组合方式。

        仅传入本次真正结束的任务；submit_schedule 返回不代表本 Member
        的任务已经结束。记录应在运行时释放前读取，或由调用方保留引用。
        """
        raise NotImplementedError("Member 子任务级奖励公式尚未实现")
