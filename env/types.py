"""仿真模块共享的实体、信道与任务标识；不依赖队列或方法实现。"""

from dataclasses import dataclass
from enum import Enum


class EntityKind(str, Enum):
    """调度中可作为通信端点的实体类别。"""

    GROUND_DEVICE = "ground_device"
    MEMBER_UAV = "member_uav"
    BS = "bs"


@dataclass(frozen=True, order=True)
class EntityRef:
    """实体类别及其类别内唯一索引。"""

    kind: EntityKind
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EntityKind(self.kind))


@dataclass(frozen=True)
class DirectedChannelKey:
    """独立信道的有向端点对。"""

    source: EntityRef
    target: EntityRef


@dataclass(frozen=True, order=True)
class TaskKey:
    owner_member_id: int
    user_id: int
    dag_id: int
    node_id: int
