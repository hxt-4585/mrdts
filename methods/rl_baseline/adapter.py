"""Default-config scene and atomic slot adapter, with no simulator monkey patches."""

from dataclasses import dataclass

import numpy as np

from env.communication.channel_model import ChannelModel
from env.contracts import DAGRequest, PlacementDecision
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User
from env.runtime.slot_result import SlotResult
from env.simulator import Simulator
from env.types import EntityKind, EntityRef
from env.workload.dag_generator import DAGGenerator
from methods.ers import ERS


def delay_metrics(result: SlotResult) -> dict[str, float]:
    """Censored mean DAG delay, counting every DAG equally (including failures)."""
    horizon = result.slot_end - result.slot_start
    if horizon <= 0 or not result.dags:
        raise ValueError('A positive slot duration and at least one DAG are required')
    delays = [horizon if dag.failed else dag.completion_time - result.slot_start
              for dag in result.dags]
    mean_delay = float(np.mean(np.clip(delays, 0., horizon)))
    return dict(mean_delay_s=mean_delay, reward=-mean_delay / horizon,
                failure_rate=sum(d.failed for d in result.dags) / len(result.dags),
                max_delay_s=float(max(delays)))


@dataclass
class PlanningBatch:
    runtime: object
    requests: tuple
    plan: object
    flight_rejected: bool
    boundary_violations: np.ndarray
    migrations: int


class DelayEnvironment:
    """Only the adapter defines the empty-region flight rejection fallback.

    reset resets positions/time, not the DAG random stream. Workloads retain the
    original default seed and parameters; an offset reserves disjoint eval slots.
    """

    def __init__(self, workload_offset: int = 0):
        if workload_offset < 0:
            raise ValueError('workload_offset must be nonnegative')
        self.region = Region()
        self.region.generate()
        self.users = User()
        self.users.generate_from_region(self.region)
        self.masters = MasterUAV()
        self.masters.generate_from_region(self.region)
        self.members = MemberUAV()
        self.members.generate_from_region_and_user(self.region, self.users, self.masters)
        self.member_count = self.members.member_uav_count
        self.master_count = self.masters.config.master_uav_count
        self.action_count = self.member_count + 2
        self.initial_positions = self.members.positions.copy()
        self.initial_regions = self.members.region_ids.copy()
        self.generator = DAGGenerator()
        self.workload_slots = 0
        for _ in range(workload_offset):
            self.generator.generate(self.users.num_users)
            self.workload_slots += 1
        self.reset()

    def reset(self):
        if hasattr(self, 'simulator') and self.simulator.runtime is not None:
            raise RuntimeError('Cannot reset during planning')
        self.members.positions[:] = self.initial_positions
        self.members.region_ids[:] = self.initial_regions
        self.members.region_member_counts = np.bincount(
            self.initial_regions[:self.member_count], minlength=self.master_count + 1)[1:]
        self.simulator = Simulator(self.members, ChannelModel(), self.users.config.transmit_power,
                                   self.users.config.core_frequency)
        self.simulator.refresh_slot_topology(self.region, self.users.positions)

    def _flight_is_feasible(self, actions):
        if actions.shape != (self.member_count, 2) or not np.isfinite(actions).all():
            raise ValueError('Flight actions must be finite with shape (member_count, 2)')
        if (np.abs(actions) > 1).any():
            raise ValueError('Flight actions must be in [-1, 1]')
        xy = self.members.positions[:self.member_count, :2].copy()
        cfg = self.members.config
        candidate = xy + actions * cfg.max_horizontal_speed * cfg.flight_duration
        side = self.region.config.side_length
        in_bounds = ((candidate >= 0.) & (candidate <= side)).all(axis=1)
        xy[in_bounds] = candidate[in_bounds]
        regions = {self.region.get_region_id(*position) for position in xy}
        return set(map(int, self.users.region_ids)).issubset(regions)

    def begin(self, actions=None) -> PlanningBatch:
        if self.simulator.runtime is not None:
            raise RuntimeError('Finish the current batch before beginning another slot')
        actions = (np.zeros((self.member_count, 2), dtype=np.float32) if actions is None
                   else np.asarray(actions, dtype=np.float32))
        rejected = not self._flight_is_feasible(actions)
        if rejected:
            actions = np.zeros_like(actions)
        old_regions = self.members.region_ids[:self.member_count].copy()
        violations = self.simulator.begin_slot(self.region, self.users.positions, actions)
        dags = self.generator.generate(self.users.num_users)
        self.workload_slots += 1
        requests = tuple(DAGRequest(0, dag,
                         EntityRef(EntityKind.MEMBER_UAV, int(self.simulator.user_member_ids[user])),
                         EntityRef(EntityKind.GROUND_DEVICE, user)) for user, dag in enumerate(dags))
        runtime = self.simulator.runtime
        plan = ERS(runtime).plan(requests)
        return PlanningBatch(runtime, requests, plan, rejected, violations,
                             int(np.count_nonzero(old_regions != self.members.region_ids[:self.member_count])))

    def execution_node(self, key, action: int) -> EntityRef:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError('Placement action must be an integer')
        if action == 0:
            return EntityRef(EntityKind.GROUND_DEVICE, key.user_id)
        if 1 <= action <= self.member_count:
            return EntityRef(EntityKind.MEMBER_UAV, int(action - 1))
        if action == self.member_count + 1:
            return EntityRef(EntityKind.BS, 0)
        raise ValueError('Placement action out of range')

    def finish(self, batch: PlanningBatch, actions: dict) -> dict[str, float]:
        if batch.runtime is not self.simulator.runtime:
            raise ValueError('Batch does not belong to the active slot')
        if set(actions) != set(batch.plan.order):
            raise ValueError('Exactly one placement per task is required')
        placements = {key: PlacementDecision(self.execution_node(key, actions[key]), seq)
                      for seq, key in enumerate(batch.plan.order)}
        batch.runtime.submit_dags(batch.requests, placements, batch.runtime.slot_start)
        metrics = delay_metrics(self.simulator.end_slot())
        metrics.update(ground_fraction=sum(a == 0 for a in actions.values()) / len(actions),
                       flight_rejected=float(batch.flight_rejected),
                       boundary_violation_rate=float(batch.boundary_violations.mean()),
                       migrations=float(batch.migrations))
        return metrics
