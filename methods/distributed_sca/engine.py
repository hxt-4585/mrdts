"""Independent frozen scheduler and local executed-traffic measurement adapter.

Only the common environment is shared with other methods. Checkpoint actor
weights are inference-only. Regional traffic statistics are observations from a
completed slot, not samples from future DAGs or a fitted high-level predictor.
"""

from collections import defaultdict
from dataclasses import replace
import heapq

import numpy as np
import torch
from torch.nn import functional as F

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.runtime.dag_runtime import topological_order
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.settings.uav_config import MemberPropulsionConfig
from env.types import EntityKind, EntityRef

from .planner import LocalProblem


def restore_settings(snapshot, slots):
    uav = dict(snapshot["uav"])
    uav["member_propulsion"] = MemberPropulsionConfig(**uav["member_propulsion"])
    return dict(region=RegionConfig(**snapshot["region"]), user=UserConfig(**snapshot["user"]),
                uav=UAVConfig(**uav), channel=ChannelConfig(**snapshot["channel"]),
                dag=DAGConfig(**snapshot["dag"]),
                env=replace(EnvConfig(**snapshot["env"]), steps_per_epoch=slots))


def ranked_orders(runtime, requests):
    """Core-weighted upward ranks with routed communication and stable topo ties."""
    ranks, indegrees, successors = {}, {}, {}
    for request in requests:
        if request.owner_member is None:
            continue
        dag, keys = request.dag, request.task_keys
        candidates = runtime.candidate_execution_nodes(request.owner_member, request.ground_device)
        core_counts = np.array([len(runtime.servers[node].core_frequencies) for node in candidates])
        total_cores = int(core_counts.sum())
        inverse_frequency = sum(1. / frequency for node in candidates
                                for frequency in runtime.servers[node].core_frequencies) / total_cores
        communication = 0.
        for source_index, source in enumerate(candidates):
            for target_index, target in enumerate(candidates):
                route = runtime.route_planner.predecessor_route(
                    source, target, request.ground_device, request.owner_member)
                communication += core_counts[source_index] * core_counts[target_index] * sum(
                    runtime.transfer_duration_s(hop, 1.) for hop in route.hops)
        communication /= total_cores**2
        children = [[] for _ in keys]
        for key in keys:
            indegrees[key] = 0
            successors[key] = []
        for parent, child in dag.edges:
            children[parent].append(child)
            successors[keys[parent]].append(keys[child])
            indegrees[keys[child]] += 1
        for node in reversed(topological_order(dag)):
            tail = max((float(dag.edge_features[node, child]) * 1000. * communication + ranks[keys[child]]
                        for child in children[node]), default=0.)
            ranks[keys[node]] = float(dag.node_features[node, 1]) * inverse_frequency + tail
    ready = [(-ranks[key], key) for key, degree in indegrees.items() if degree == 0]
    heapq.heapify(ready)
    orders = defaultdict(list)
    while ready:
        _, key = heapq.heappop(ready)
        orders[key.owner_member_id].append(key)
        for child in successors[key]:
            indegrees[child] -= 1
            if indegrees[child] == 0:
                heapq.heappush(ready, (-ranks[child], child))
    return dict(orders)


class Simulator:
    def __init__(self, source, slots):
        self.env = MRDTSEnv(restore_settings(source["environment"], slots))
        self.env.generate_scene()
        state = source["member_policies"]
        self.actors = tuple(tuple((state[f"{member}.actor.{layer}.weight"].detach(),
                                   state[f"{member}.actor.{layer}.bias"].detach()) for layer in (0, 2, 4))
                            for member in range(self.env.members.member_uav_count))

    def choose(self, key):
        env = self.env
        count = env.members.member_uav_count
        full = env.observe_member(key)
        # This fixed source checkpoint predates the M energy-queue observations.
        observation = np.concatenate((full[:4 + 4 * count], full[4 + 5 * count:])).astype(np.float32)
        hidden = torch.from_numpy(observation)
        actor = self.actors[key.owner_member_id]
        for weight, bias in actor[:-1]:
            hidden = torch.tanh(F.linear(hidden, weight, bias))
        logits = F.linear(hidden, *actor[-1]).numpy()
        region = env.members.region_ids[key.owner_member_id]
        available = np.r_[np.flatnonzero(env.members.region_ids[:count] == region), count, count + 1]
        action = int(available[np.argmax(logits[available])])
        if action < count:
            target = EntityRef(EntityKind.MEMBER_UAV, action)
        elif action == count:
            target = EntityRef(EntityKind.BS, 0)
        else:
            target = EntityRef(EntityKind.GROUND_DEVICE, key.user_id)
        return PlacementDecision(target)

    @torch.inference_mode()
    def step(self, actions):
        beginning = self.env.begin_step(actions)
        runtime = self.env._runtime
        orders = ranked_orders(runtime, beginning.requests)
        cursors = dict.fromkeys(orders, 0)
        ready = beginning.ready_member_ids
        while ready:
            decisions = {}
            for member in ready:
                key = orders[member][cursors[member]]
                decisions[key] = self.choose(key)
                cursors[member] += 1
            ready = self.env.submit_schedule(decisions)
        traffic = collect_traffic(self.env, runtime)
        return self.env.end_step(), traffic


def collect_traffic(env, runtime):
    """Aggregate only each region's executed local transfers.

    bits_eff = bits * (observed service time / isolated-link service time)
               * unfinished DAG count / total user count.
    The resource-sharing ratio is measurable at the link endpoints; no remote
    activity, global channel count, or remote scheduling state is supplied to the
    planner. Summing route-hop service times approximates parallel data arrivals.
    """
    traffic = [defaultdict(float) for _ in range(env.masters.num_uavs)]
    for key, task in runtime.tasks.items():
        if key.owner_member_id == -1:
            continue
        master = int(env.users.region_ids[key.user_id]) - 1
        factor = task.unfinished_dags_at_decision / env.users.num_users
        records = (task.input_record, *task.predecessor_records.values())
        for record in records:
            for transfer_id in record.transfer_ids:
                job = runtime.transfer_jobs[transfer_id]
                hop = runtime._transfer_channel[transfer_id]
                observed_s = job.finish_at - job.start_at
                reference_s = runtime.transfer_duration_s(hop, job.payload_bits)
                traffic[master][hop.source, hop.target] += job.payload_bits * observed_s / reference_s * factor
    return [dict(links) for links in traffic]


def build_problem(env, master, traffic, prior_weight, smoothing_m):
    """Compact local frame: own users/traffic, one-hop UAVs, public fixed BS.

    A small analytical expected-input prior supplies delay terms for currently
    idle UAVs. Six local Lloyd steps only allocate this prior's service zones;
    they never issue a flight command. The prior uses configured workload means
    and a fixed 1/2 relay fraction, not future tasks or a learned model.
    """
    region = int(env.masters.region_ids[master])
    visible_regions = (region, *env.masters.neighbor_region_ids[region])
    count = env.members.member_uav_count
    regions = env.members.region_ids[:count]
    visible = np.flatnonzero(np.isin(regions, visible_regions))
    owned = np.flatnonzero(regions == region)
    users = np.flatnonzero(env.users.region_ids == region)
    member_nodes = [EntityRef(EntityKind.MEMBER_UAV, int(i)) for i in visible]
    user_nodes = [EntityRef(EntityKind.GROUND_DEVICE, int(i)) for i in users]
    bs = EntityRef(EntityKind.BS, 0)
    entities = [*member_nodes, *user_nodes, bs]
    rows = {node: index for index, node in enumerate(entities)}
    positions = np.concatenate((env.members.positions[visible], env.users.positions[users],
                                env.members.positions[count:count + 1])).astype(float)
    links = defaultdict(float)
    if traffic is not None:
        for (source, target), bits in traffic.items():
            # Ownership handoff may make an old endpoint no longer observable.
            if source in rows and target in rows:
                links[source, target] += (1. - prior_weight) * bits
    if len(owned) and len(users):
        centers = env.members.positions[owned, :2].astype(float)
        user_xy = env.users.positions[users, :2]
        for _ in range(6):
            assignment = np.argmin(np.sum((user_xy[:, None] - centers[None, :])**2, axis=2), axis=1)
            for index in range(len(owned)):
                selected = assignment == index
                if np.any(selected):
                    centers[index] = user_xy[selected].mean(axis=0)
        assignment = np.argmin(np.sum((user_xy[:, None] - centers[None, :])**2, axis=2), axis=1)
        mean_input_bits = env.settings["dag"].n * sum(env.settings["dag"].input_data_range) * 500.
        mean_unfinished = .5 * (1. + len(users) / len(owned))
        prior_bits = mean_input_bits * mean_unfinished / env.users.num_users
        prior_bits *= 1. if traffic is None else prior_weight
        for user_node, index in zip(user_nodes, assignment):
            member_node = EntityRef(EntityKind.MEMBER_UAV, int(owned[index]))
            links[user_node, member_node] += prior_bits
            links[member_node, bs] += .5 * prior_bits
    sources, targets, bits, powers, airborne = [], [], [], [], []
    for (source, target), weight in links.items():
        sources.append(rows[source])
        targets.append(rows[target])
        bits.append(weight)
        powers.append(env.users.config.transmit_power if source.kind is EntityKind.GROUND_DEVICE else
                      env.members.config.bs_transmit_power if source.kind is EntityKind.BS else
                      env.members.config.member_transmit_power)
        airborne.append(source.kind is EntityKind.MEMBER_UAV and target.kind is EntityKind.MEMBER_UAV)
    return LocalProblem(visible, owned, positions, np.array(sources), np.array(targets),
                        np.array(bits), np.array(powers), np.array(airborne), env.settings["channel"], smoothing_m)
