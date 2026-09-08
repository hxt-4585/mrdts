"""Local actor features and explicitly separated centralized critic context."""

from collections import defaultdict

import numpy as np


MEMBER_FEATURES = 23


def central_state(env, batch=None):
    """Training-only state: positions, ownership, user counts and workload summary."""
    count = env.member_count
    owners = env.simulator.user_member_ids
    xy = env.members.positions[:count, :2] / env.region.config.side_length
    regions = env.members.region_ids[:count, None] / env.master_count
    users = np.bincount(owners, minlength=count)[:, None] / env.users.num_users
    workload = np.zeros(3)
    if batch is not None:
        features = np.concatenate([r.dag.node_features for r in batch.requests])
        workload = [features[:, 0].mean() / 500., features[:, 1].mean() / 1e8,
                    len(batch.plan.order) / 1000.]
    return np.concatenate([np.column_stack([xy, regions, users]).ravel(), workload]).astype(np.float32)


def master_observations(env):
    """One local-region observation per Master, with fixed physical Member IDs."""
    observations, masks = [], []
    side = env.region.config.side_length
    xy = env.members.positions[:env.member_count, :2] / side
    user_counts = np.bincount(env.simulator.user_member_ids, minlength=env.member_count)
    for region in range(1, env.master_count + 1):
        owned = env.members.region_ids[:env.member_count] == region
        user_xy = env.users.positions[env.users.region_ids == region, :2] / side
        center = user_xy.mean(axis=0) if len(user_xy) else np.zeros(2)
        spread = user_xy.std(axis=0) if len(user_xy) else np.zeros(2)
        local = np.column_stack([xy, center - xy, user_counts / env.users.num_users])
        local[~owned] = 0.
        context = [*center, *spread, len(user_xy) / env.users.num_users,
                   owned.mean(), * (env.masters.positions[region - 1, :2] / side)]
        observations.append(np.concatenate([local.ravel(), owned, context]))
        masks.append(np.repeat(owned, 2))
    return np.asarray(observations, dtype=np.float32), np.asarray(masks, dtype=bool)


class MemberPlanning:
    """Observe only own DAGs, known resources and own earlier assignments.

    Feature construction never submits tasks or reads other Members' plans. Local
    planned work is a proxy, not a measurement of shared queue occupancy.
    """

    def __init__(self, env, batch):
        self.env, self.batch = env, batch
        self.requests = {r.key: r for r in batch.requests}
        self.placements = {}
        self.planned_work = defaultdict(float)
        self.done = defaultdict(int)
        self.parents = {r.key: {node: [(a, r.dag.edge_features[a, b]) for a, b in r.dag.edges
                                      if b == node] for node in range(r.dag.node_num)}
                        for r in batch.requests}
        self.base = {}
        self.masks = {}
        for request in batch.requests:
            rows = np.zeros((env.action_count, 12), dtype=np.float32)
            mask = np.zeros(env.action_count, dtype=bool)
            costs = batch.plan.costs[request.key]
            for candidate in costs.candidates:
                if candidate == request.ground_device:
                    action, kind = 0, 0
                elif candidate == batch.runtime.global_bs:
                    action, kind = env.action_count - 1, 2
                else:
                    action, kind = candidate.index + 1, 1
                mask[action] = True
                freq = batch.runtime.servers[candidate].core_frequencies
                route = batch.runtime.route_planner.input_route(
                    request.ground_device, request.owner_member, candidate)
                seconds_bit = sum(batch.runtime.transfer_duration_s(hop, 1.) for hop in route.hops)
                rows[action, kind] = 1.
                rows[action, 3] = float(candidate == request.owner_member)
                rows[action, 4] = 1e8 / max(freq)
                rows[action, 5] = len(freq) / 4.
                rows[action, 6] = seconds_bit * 1000. * 500.
                rows[action, 7] = np.linalg.norm(batch.runtime.entity_positions[candidate][:2] -
                                               env.users.positions[request.ground_device.index, :2]) / env.region.config.side_length
            self.base[request.key], self.masks[request.key] = rows, mask

    def observe(self, key):
        request = self.requests[key.owner_member_id, key.user_id, key.dag_id]
        dag = request.dag
        member_order = self.batch.plan.member_orders[request.owner_member]
        owner = key.owner_member_id
        node = key.node_id
        parents = self.parents[request.key][node]
        context = np.array([
            dag.node_features[node, 0] / 500., dag.node_features[node, 1] / 1e8,
            self.batch.plan.ranks[key], len(parents) / dag.node_num,
            sum(a == node for a, _ in dag.edges) / dag.node_num,
            dag.node_features[:, 1].sum() / 1e9,
            self.done[owner] / len(member_order), len(member_order) / 1000.,
            np.linalg.norm(self.env.members.positions[owner, :2] - self.env.users.positions[key.user_id, :2]) /
            self.env.region.config.side_length,
            node / dag.node_num, dag.node_num / 10.], dtype=np.float32)
        rows = self.base[request.key].copy()
        mask = self.masks[request.key]
        costs = self.batch.plan.costs[request.key]
        for action in np.flatnonzero(mask):
            candidate = self.env.execution_node(key, int(action))
            rows[action, 4] *= context[1]
            rows[action, 6] *= context[0]
            rows[action, 8] = self.planned_work[owner, candidate]
            comm, colocated = 0., 0
            for parent, kbit in parents:
                parent_key = type(key)(owner, key.user_id, key.dag_id, parent)
                source = self.env.execution_node(parent_key, self.placements[parent_key])
                comm += costs.path_seconds_per_bit[source, candidate] * kbit * 1000.
                colocated += source == candidate
            rows[action, 9] = comm
            rows[action, 10] = colocated / max(1, len(parents))
            rows[action, 11] = self.done[owner] / len(member_order)
        observation = np.concatenate([np.broadcast_to(context, (self.env.action_count, len(context))), rows], axis=1)
        # Bounded features keep the MLP stable; rewards use uncropped runtime values.
        return np.clip(observation, -10., 10.).astype(np.float32), mask.copy()

    def assign(self, key, action):
        request = self.requests[key.owner_member_id, key.user_id, key.dag_id]
        if key in self.placements:
            raise ValueError('Task already assigned')
        if not 0 <= action < self.env.action_count or not self.masks[request.key][action]:
            raise ValueError('Illegal execution candidate')
        candidate = self.env.execution_node(key, int(action))
        frequency = sum(self.batch.runtime.servers[candidate].core_frequencies)
        self.planned_work[key.owner_member_id, candidate] += request.dag.node_features[key.node_id, 1] / frequency
        self.placements[key] = int(action)
        self.done[key.owner_member_id] += 1
