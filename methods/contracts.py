"""算法业务接口；不要求组件具有 RL 观测、奖励或训练能力。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

import numpy as np

from env.contracts import DAGRequest
from env.runtime.event_runtime import SchedulingRuntime
from env.types import EntityRef, TaskKey

if TYPE_CHECKING:
    import torch
    from experiments.config import ExperimentConfig


class OrderingPlan(Protocol):
    @property
    def order(self) -> tuple[TaskKey, ...]: ...


class OrderingComponent(Protocol):
    def plan(self, runtime: SchedulingRuntime, requests: Sequence[DAGRequest]) -> OrderingPlan:
        """只读资源与通信成本，不推进或提交真实运行时。"""
        ...


@dataclass(frozen=True)
class FlightContext:
    master_id: int
    member_ids: tuple[int, ...]
    positions: np.ndarray


@dataclass(frozen=True)
class FlightDecision:
    member_actions: Mapping[int, tuple[float, float]]


class FlightComponent(Protocol):
    def decide(self, context: FlightContext) -> FlightDecision: ...


@dataclass(frozen=True)
class SchedulingContext:
    owner: EntityRef
    requests: tuple[DAGRequest, ...]
    order: tuple[TaskKey, ...]
    candidates: Mapping[TaskKey, tuple[EntityRef, ...]]


class SchedulingComponent(Protocol):
    def schedule(self, context: SchedulingContext) -> Mapping[TaskKey, EntityRef]: ...


class Trainer(Protocol):
    def train(self, config: ExperimentConfig, device: torch.device) -> Path:
        """方法自己的训练组织；返回模型和训练记录所在的运行目录。"""
        ...
