"""时隙结束后的不可变结果，不持有运行时或队列。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MemberEnergyResult:
    member_id: int
    compute_energy_j: float
    tx_energy_j: float


@dataclass(frozen=True)
class DAGResult:
    key: tuple[int, int, int]  # owner Member、用户、DAG 编号
    completion_time: float | None
    failed: bool


@dataclass(frozen=True)
class SlotResult:
    """计算和发送能耗仅统计 Member UAV，不包含 BS、用户或推进能耗。"""

    slot_start: float
    slot_end: float
    dags: tuple[DAGResult, ...]
    compute_energy_j: float
    tx_energy_j: float
    member_energy: tuple[MemberEnergyResult, ...]

    @property
    def finished_dags(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(dag.key for dag in self.dags if not dag.failed)

    @property
    def failed_dags(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(dag.key for dag in self.dags if dag.failed)
