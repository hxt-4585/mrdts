"""通用时隙流程；特殊联合优化方法可以提供自己的完整流程。"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from env.contracts import DAGRequest, PlacementDecision
from env.runtime.slot_result import SlotResult
from env.types import EntityKind, EntityRef
from methods.contracts import (FlightComponent, FlightContext, OrderingComponent,
                               SchedulingComponent, SchedulingContext)


@dataclass(frozen=True)
class SlotOutcome:
    result: SlotResult
    flight_violations: int
    flight_resamples: int = 0
    flight_fallbacks: int = 0


class CompositeMethod:
    def __init__(self, ordering: OrderingComponent, flight: FlightComponent,
                 scheduling: SchedulingComponent):
        self.ordering = ordering
        self.flight = flight
        self.scheduling = scheduling
        self.flight_resamples = 0
        self.flight_fallbacks = 0

    def flight_contexts(self, scene):
        """完整方法决定信息范围；此参照只向 Master 提供原所属 Member 位置。"""
        members = scene.simulator.members
        for master_id, region_id in enumerate(scene.masters.region_ids):
            ids = tuple(int(i) for i in scene.simulator.member_ids_in_region(int(region_id)))
            positions = members.positions[list(ids)].copy()
            positions.setflags(write=False)
            yield FlightContext(master_id, ids, positions)

    def scheduling_context(self, owner, requests, order, runtime):
        owned = tuple(request for request in requests if request.owner_member == owner)
        candidates = {key: runtime.candidate_execution_nodes(owner, request.ground_device)
                      for request in owned for key in request.task_keys}
        return SchedulingContext(owner, owned,
                                 tuple(key for key in order if key.owner_member_id == owner.index),
                                 MappingProxyType(candidates))

    def flight_actions(self, scene):
        """收集各 Master 的动作，不修改仿真状态。"""
        simulator = scene.simulator
        actions = {}
        for context in self.flight_contexts(scene):
            decision = self.flight.decide(context)
            if set(decision.member_actions) != set(context.member_ids):
                raise ValueError("Flight component must provide exactly its Master's Member actions")
            actions.update(decision.member_actions)
        count = simulator.members.member_uav_count
        if set(actions) != set(range(count)):
            raise ValueError("Flight decisions must cover every Member")
        return np.asarray([actions[i] for i in range(count)], dtype=float)

    def begin_slot(self, scene):
        return scene.simulator.begin_slot(scene.region, scene.users.positions, self.flight_actions(scene))

    def run_slot(self, scene, workload):
        """workload 为 (dag_id, user_id, DAG) 序列；移动后重新确定 owner。"""
        simulator = scene.simulator
        violations = self.begin_slot(scene)
        runtime = simulator.runtime
        requests = tuple(DAGRequest(dag_id, dag,
                         EntityRef(EntityKind.MEMBER_UAV, int(simulator.user_member_ids[user_id])),
                         EntityRef(EntityKind.GROUND_DEVICE, user_id))
                         for dag_id, user_id, dag in workload)
        plan = self.ordering.plan(runtime, requests)
        required = {key for request in requests for key in request.task_keys}
        if len(plan.order) != len(required) or set(plan.order) != required:
            raise ValueError("Ordering must contain every task exactly once")
        placements = {}
        for owner in sorted({request.owner_member for request in requests}):
            context = self.scheduling_context(owner, requests, plan.order, runtime)
            decision = self.scheduling.schedule(context)
            if set(decision) != set(context.order):
                raise ValueError("Scheduling must place every owned task exactly once")
            if any(node not in context.candidates[key] for key, node in decision.items()):
                raise ValueError("Scheduling selected an illegal execution node")
            placements.update(decision)
        runtime.submit_dags(requests, {key: PlacementDecision(placements[key], seq)
                                      for seq, key in enumerate(plan.order)}, runtime.slot_start)
        return SlotOutcome(simulator.end_slot(), int(np.count_nonzero(violations)),
                           self.flight_resamples, self.flight_fallbacks)
