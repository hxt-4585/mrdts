"""方法组件与仿真运行器之间的稳定契约。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from env.dag_generator import DAG
from env.dag_runtime import DAGRuntime


@dataclass(frozen=True)
class FlightDecision:
    """一个时隙内全部 Member UAV 的二维飞行动作。"""

    member_actions: np.ndarray


@dataclass(frozen=True)
class SchedulingRequest:
    """一次 DAG 调度所需的、与当前环境实现无关的输入。"""

    user_position: np.ndarray
    user_region_id: int


@dataclass(frozen=True)
class TaskAssignment:
    """一个子任务的执行位置与时间安排。"""

    node_id: int
    server_id: int
    start_time: float
    finish_time: float
    ingress_relay_id: int | None = None


@dataclass(frozen=True)
class DAGSchedulingResult:
    """单个 DAG 的调度结果。"""

    assignments: tuple[TaskAssignment, ...]
    completion_time: float


class TrajectoryMethod(Protocol):
    """可替换的 Member 飞行决策组件。"""

    def decide(self, member_count: int) -> FlightDecision: ...


class SchedulingMethod(Protocol):
    """可替换的 DAG 调度组件。"""

    def schedule_dag(
        self, dag: DAG, runtime: DAGRuntime, request: SchedulingRequest
    ) -> DAGSchedulingResult: ...
