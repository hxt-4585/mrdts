"""PPO planning on the current Scene; one execution path for training and inference."""

from copy import copy
from dataclasses import dataclass

import numpy as np

from env.contracts import DAGRequest, PlacementDecision
from env.types import EntityKind, EntityRef
from methods.compose import SlotOutcome
from methods.components.ordering.adapter import ERSOrdering
from .reward import delay_metrics


@dataclass
class PlanningBatch:
    runtime: object
    requests: tuple
    plan: object
    flight_rejected: bool
    boundary_violations: np.ndarray
    migrations: int


class PlanningEnvironment:
    """A view of an experiment-owned scene; does not generate workloads or RNGs."""

    def __init__(self, scene, ordering=None):
        self.scene = scene
        self.region, self.users, self.masters = scene.region, scene.users, scene.masters
        self.simulator = scene.simulator
        self.members = self.simulator.members
        self.member_count = self.members.member_uav_count
        self.master_count = self.masters.config.master_uav_count
        self.action_count = self.member_count + 2
        self.ordering = ordering or ERSOrdering()
        if self.simulator.user_member_ids is None:
            self.simulator.refresh_slot_topology(self.region, self.users.positions)

    def begin(self, workload, actions):
        if self.simulator.runtime is not None:
            raise RuntimeError('Finish the active slot before beginning another')
        # Preview through the physical implementation, including per-UAV boundary rejection.
        preview = copy(self.members)
        preview.positions = self.members.positions.copy()
        preview.apply_flight_actions(actions, self.region.config.side_length)
        occupied = {self.region.get_region_id(*position[:2])
                    for position in preview.positions[:self.member_count]}
        rejected = not set(map(int, self.users.region_ids)).issubset(occupied)
        executed = np.zeros_like(actions) if rejected else actions
        before = self.members.region_ids[:self.member_count].copy()
        violations = self.simulator.begin_slot(self.region, self.users.positions, executed)
        requests = tuple(DAGRequest(dag_id, dag,
                         EntityRef(EntityKind.MEMBER_UAV, int(self.simulator.user_member_ids[user])),
                         EntityRef(EntityKind.GROUND_DEVICE, user)) for dag_id, user, dag in workload)
        if len(requests) != self.users.num_users or {r.ground_device.index for r in requests} != set(range(self.users.num_users)):
            raise ValueError('Every user must generate exactly one DAG per slot')
        runtime = self.simulator.runtime
        plan = self.ordering.plan(runtime, requests)
        required = {key for request in requests for key in request.task_keys}
        if len(plan.order) != len(required) or set(plan.order) != required:
            raise ValueError('Ordering must cover every generated task exactly once')
        return PlanningBatch(runtime, requests, plan, rejected, violations,
                             int(np.count_nonzero(before != self.members.region_ids[:self.member_count])))

    def execution_node(self, key, action):
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError('Placement action must be an integer')
        if action == 0:
            return EntityRef(EntityKind.GROUND_DEVICE, key.user_id)
        if 1 <= action <= self.member_count:
            return EntityRef(EntityKind.MEMBER_UAV, int(action - 1))
        if action == self.member_count + 1:
            return EntityRef(EntityKind.BS, 0)
        raise ValueError('Placement action out of range')

    def finish(self, batch, actions):
        if batch.runtime is not self.simulator.runtime:
            raise ValueError('Batch does not belong to the active slot')
        if set(actions) != set(batch.plan.order):
            raise ValueError('Exactly one placement per generated task is required')
        placements = {key: PlacementDecision(self.execution_node(key, actions[key]), seq)
                      for seq, key in enumerate(batch.plan.order)}
        batch.runtime.submit_dags(batch.requests, placements, batch.runtime.slot_start)
        outcome = SlotOutcome(self.simulator.end_slot(), int(batch.boundary_violations.sum()),
                              flight_fallbacks=int(batch.flight_rejected))
        metrics = delay_metrics(outcome.result)
        metrics.update(ground_fraction=sum(a == 0 for a in actions.values()) / len(actions),
                       flight_rejected=float(batch.flight_rejected),
                       boundary_violation_rate=float(batch.boundary_violations.mean()),
                       migrations=float(batch.migrations))
        return outcome, metrics


class PPOMethod:
    def __init__(self, learner=None, *, checkpoint=None, device='cpu', ordering=None, flight=None, scheduling=None):
        self.learner = learner
        self.checkpoint = checkpoint
        self.device = device
        self.ordering = ordering or ERSOrdering()
        self.flight, self.scheduling = flight, scheduling
        self.last_metrics = {}

    def execute(self, scene, workload, *, stage='evaluation', remaining=1.):
        from .rollout import Learner, choose_master, plan_members
        env = PlanningEnvironment(scene, self.ordering)
        if self.learner is None:
            if self.checkpoint is None:
                raise ValueError('PPO evaluation requires --checkpoint (train via experiments.train)')
            from .checkpoint import read_checkpoint
            state = read_checkpoint(self.checkpoint)
            self.learner = Learner(env, hidden=state['hidden'], device=self.device)
            self.learner.load(self.checkpoint, optimizers=False, restore_rng=False)
            if self.flight is not None:
                self.flight.actor = self.learner.master
                self.learner.flight = self.flight
            if self.scheduling is not None:
                self.scheduling.actor = self.learner.member
                self.learner.scheduling = self.scheduling
        flight, master_chunk = choose_master(env, self.learner, remaining, deterministic=stage != 'master')
        batch = env.begin(workload, flight)
        actions, member_chunks = plan_members(env, batch, self.learner,
                                             deterministic=stage != 'member', collect=stage == 'member')
        outcome, metrics = env.finish(batch, actions)
        self.last_metrics = metrics
        return outcome, metrics, member_chunks, master_chunk

    def run_slot(self, scene, workload):
        return self.execute(scene, workload)[0]


def build_method(ordering, flight, scheduling):
    return PPOMethod(ordering=ordering, flight=flight, scheduling=scheduling)
