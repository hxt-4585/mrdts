"""逐核心、中继路径感知的 ERS 时延排序；不选择任务执行位置。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np

from env.contracts import DAGRequest
from env.runtime.dag_runtime import topological_order, validate_dag
from env.types import DirectedChannelKey, EntityRef, TaskKey
from env.workload.dag_generator import DAG

if TYPE_CHECKING:
    from env.runtime.event_runtime import SchedulingRuntime


@dataclass(frozen=True)
class AverageCosts:
    """可检查的排序成本；路径单位 s/bit，其余单位 s，不包含排队或自身输入。"""

    candidates: tuple[EntityRef, ...]
    path_seconds_per_bit: dict[tuple[EntityRef, EntityRef], float]
    average_compute_s: dict[int, float]
    average_edge_comm_s: dict[tuple[int, int], float]


@dataclass(frozen=True)
class ERSPlan:
    order: tuple[TaskKey, ...]
    ranks: dict[TaskKey, float]
    member_orders: dict[EntityRef, tuple[TaskKey, ...]]
    costs: dict[tuple[int, int, int], AverageCosts]


@dataclass(frozen=True)
class _LocationCosts:
    candidates: tuple[EntityRef, ...]
    paths: dict[tuple[EntityRef, EntityRef], float]
    seconds_per_cycle: float
    seconds_per_bit: float


class ERS:
    """按核心等权估计时间，在每个 Member 的全部 DAG 中统一排序。

    对每对执行设备累加合法路径各跳的时间，再按 K_p*K_q 加权，
    分母包含同设备的零通信组合。自身输入上传仅由运行时处理。
    每次 plan 重建成本缓存，避免沿用旧位置、带宽或用户关联。
    """

    def __init__(self, runtime: SchedulingRuntime):
        self.runtime = runtime

    @staticmethod
    def ranks(
        dag: DAG, average_compute_s: Mapping[int, float],
        average_edge_comm_s: Mapping[tuple[int, int], float],
    ) -> dict[int, float]:
        order = topological_order(dag)
        if set(average_compute_s) != set(order) or set(average_edge_comm_s) != set(dag.edges):
            raise ValueError("平均成本必须覆盖 DAG 的全部节点和边")
        compute = {node: float(value) for node, value in average_compute_s.items()}
        comm = {edge: float(value) for edge, value in average_edge_comm_s.items()}
        if any(not math.isfinite(value) or value < 0 for value in (*compute.values(), *comm.values())):
            raise ValueError("平均时延必须为有限非负数")
        successors = {node: [] for node in order}
        for parent, child in dag.edges:
            successors[parent].append(child)
        values = {}
        for node in reversed(order):
            values[node] = compute[node] + max(
                (comm[node, child] + values[child] for child in successors[node]), default=0.)
            if not math.isfinite(values[node]):
                raise ValueError("rank 时延溢出")
        return {node: values[node] for node in range(dag.node_num)}

    @classmethod
    def order(
        cls, dag: DAG, average_compute_s: Mapping[int, float],
        average_edge_comm_s: Mapping[tuple[int, int], float],
    ) -> tuple[int, ...]:
        ranks = cls.ranks(dag, average_compute_s, average_edge_comm_s)
        positions = {node: i for i, node in enumerate(topological_order(dag))}
        return tuple(sorted(ranks, key=lambda node: (-ranks[node], positions[node])))

    def average_costs(self, request: DAGRequest) -> AverageCosts:
        return self._task_costs(request.dag, self._location_costs(request, {}))

    def plan(self, requests: Iterable[DAGRequest]) -> ERSPlan:
        requests = tuple(requests)
        if len({request.key for request in requests}) != len(requests):
            raise ValueError("同一 owner/user/dag_id 的 DAG 重复")
        ranks, costs, topo_positions = {}, {}, {}
        members: set[EntityRef] = set()
        locations: dict[tuple[EntityRef, EntityRef], _LocationCosts] = {}
        hop_seconds: dict[DirectedChannelKey, float] = {}
        for request in sorted(requests, key=lambda item: item.key):
            context = request.owner_member, request.ground_device
            if context not in locations:
                locations[context] = self._location_costs(request, hop_seconds)
            costs[request.key] = average = self._task_costs(request.dag, locations[context])
            node_ranks = self.ranks(request.dag, average.average_compute_s, average.average_edge_comm_s)
            for index, node in enumerate(topological_order(request.dag)):
                key = TaskKey(*request.key, node)
                ranks[key] = node_ranks[node]
                topo_positions[key] = index
            members.add(request.owner_member)
        # 全局稳定合并保留每个 Member 内部顺序，供共享 BS 和信道使用。
        order = tuple(sorted(ranks, key=lambda key: (
            -ranks[key], key.owner_member_id, key.user_id, key.dag_id, topo_positions[key])))
        member_orders = {member: tuple(key for key in order if key.owner_member_id == member.index)
                         for member in sorted(members)}
        return ERSPlan(order, ranks, member_orders, costs)

    def _location_costs(
        self, request: DAGRequest, hop_seconds: dict[DirectedChannelKey, float],
    ) -> _LocationCosts:
        candidates = self.runtime.candidate_execution_nodes(request.owner_member, request.ground_device)
        frequencies = {p: self.runtime.servers[p].core_frequencies for p in candidates}
        if any(not math.isfinite(f) or f <= 0 for cores in frequencies.values() for f in cores):
            raise ValueError("核心频率必须为有限正数")
        core_count = sum(len(cores) for cores in frequencies.values())
        seconds_per_cycle = math.fsum(1 / f for cores in frequencies.values() for f in cores) / core_count
        paths = {}
        for source in candidates:
            for target in candidates:
                route = self.runtime.route_planner.predecessor_route(
                    source, target, request.ground_device, request.owner_member)
                for hop in route.hops:
                    if hop not in hop_seconds:
                        hop_seconds[hop] = self.runtime.transfer_duration_s(hop, 1.)
                paths[source, target] = math.fsum(hop_seconds[hop] for hop in route.hops)
        seconds_per_bit = math.fsum(
            len(frequencies[p]) * len(frequencies[q]) * paths[p, q]
            for p in candidates for q in candidates) / core_count**2
        return _LocationCosts(candidates, paths, seconds_per_cycle, seconds_per_bit)

    @staticmethod
    def _task_costs(dag: DAG, locations: _LocationCosts) -> AverageCosts:
        validate_dag(dag)
        features = np.asarray(dag.node_features, dtype=float)
        return AverageCosts(
            locations.candidates, dict(locations.paths),
            {node: float(features[node, 1]) * locations.seconds_per_cycle for node in range(dag.node_num)},
            {edge: float(size) * 8192. * locations.seconds_per_bit for edge, size in dag.edge_features.items()})
