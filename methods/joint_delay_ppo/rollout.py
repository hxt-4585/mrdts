"""ERS ordering and event-driven task placement for one completed slot."""

import math

import numpy as np

from env.contracts import PlacementDecision
from env.runtime.dag_runtime import topological_order
from env.types import EntityKind, EntityRef, TaskKey
from methods.joint_delay_ppo.learning import Decision, finish_trajectory


def rank_orders(runtime, requests):
    """Rank upward with core-weighted execution and routed transmission time."""
    locations, hops, ranked = {}, {}, []
    for request in sorted(requests, key=lambda request: request.key):
        owner = request.owner_member
        if owner is None:
            continue
        context = owner, request.ground_device
        if context not in locations:
            nodes = runtime.candidate_execution_nodes(owner, request.ground_device)
            frequencies = [runtime.servers[node].core_frequencies for node in nodes]
            core_count = sum(map(len, frequencies))
            compute_factor = math.fsum(1. / float(frequency)
                for cores in frequencies for frequency in cores) / core_count
            communication_cost = 0.
            for source, source_cores in zip(nodes, frequencies):
                for destination, destination_cores in zip(nodes, frequencies):
                    route = runtime.route_planner.predecessor_route(
                        source, destination, request.ground_device, owner)
                    duration = 0.
                    for hop in route.hops:
                        if hop not in hops:
                            hops[hop] = runtime.transfer_duration_s(hop, 1.)
                        duration += hops[hop]
                    communication_cost += len(source_cores) * len(destination_cores) * duration
            locations[context] = compute_factor, communication_cost / core_count**2
        compute_factor, communication_factor = locations[context]
        dag = request.dag
        topology = topological_order(dag)
        successors = [[] for _ in range(dag.node_num)]
        for source, destination in dag.edges:
            successors[source].append(destination)
        ranks = np.empty(dag.node_num, dtype=np.float64)
        for node in reversed(topology):
            downstream = max((ranks[child] + dag.edge_features[node, child] * 1000. * communication_factor
                              for child in successors[node]), default=0.)
            ranks[node] = float(dag.node_features[node, 1]) * compute_factor + downstream
        for position, node in enumerate(topology):
            key = TaskKey(*request.key, node)
            ranked.append((-ranks[node], key.owner_member_id, key.user_id, key.dag_id, position, key))
    orders = {}
    for *_, key in sorted(ranked):
        orders.setdefault(key.owner_member_id, []).append(key)
    return orders


def collect_slot(env, model, actions, config, deterministic=False):
    count = env.members.member_uav_count
    trajectory = {member: [] for member in range(count)}
    begin = env.begin_step(actions)
    orders = {owner: iter(keys) for owner, keys in rank_orders(env._runtime, begin.requests).items()}
    regions = env.members.region_ids[:count]
    masks = regions[:, None] == regions[None, :]
    pending = {}
    ready = begin.ready_member_ids
    while ready:
        # All observations are taken before submitting any action in the same batch.
        inputs = []
        for member in ready:
            key = next(orders[member])
            raw = env.observe_member(key)
            observation = np.empty(5 * count + 8, dtype=np.float32)
            observation[:4 + 4 * count] = raw[:4 + 4 * count]
            observation[4 + 4 * count:] = raw[4 + 5 * count:]
            mask = np.ones(count + 2, dtype=bool)
            mask[:count] = masks[member]
            inputs.append((member, key, observation, mask))
        placements = {}
        for member, key, observation, mask in inputs:
            action, logp, value = model[member].act(observation, mask, deterministic)
            sample = Decision(observation, mask, action, logp, value, key=key)
            trajectory[member].append(sample)
            pending[key] = sample
            if action < count:
                endpoint = EntityRef(EntityKind.MEMBER_UAV, action)
            elif action == count:
                endpoint = EntityRef(EntityKind.BS, 0)
            else:
                endpoint = EntityRef(EntityKind.GROUND_DEVICE, key.user_id)
            placements[key] = PlacementDecision(endpoint)
        ready = env.submit_schedule(placements)
        for key, reward in env.pop_member_rewards().items():
            pending.pop(key).reward = reward
    result = env.end_step()
    for key, reward in result.member_rewards.items():
        pending.pop(key).reward = reward
    for samples in trajectory.values():
        finish_trajectory(samples, config['member_gamma'], config['gae_lambda'])
    slot = result.slot_result
    regional_delays = np.zeros(env.region.config.region_count, dtype=np.float64)
    failed = 0
    successful_delay = 0.
    for dag in slot.dags:
        delay = dag.completion_time - slot.slot_start
        regional_delays[int(env.users.region_ids[dag.key[1]]) - 1] += delay
        failed += int(dag.failed)
        if not dag.failed:
            successful_delay += delay
    dags = len(slot.dags)
    delay_sum = float(regional_delays.sum())
    decisions = sum(map(len, trajectory.values()))
    row = dict(episode=env.epoch_index, slot=env.step_index - 1, delay_sum_s=delay_sum,
               dag_count=dags, failed_dags=failed, flight_violations=int(begin.flight_violations.sum()),
               member_decisions=decisions,
               member_reward_sum=sum(sample.reward for samples in trajectory.values() for sample in samples),
               mean_delay_s=delay_sum / dags,
               failure_rate=failed / dags,
               successful_delay_sum_s=successful_delay,
               successful_mean_delay_s=successful_delay / (dags - failed) if dags > failed else None,
               global_delay_reward=result.master_reward, total_energy_j=float(result.total_energy_j.sum()),
               mean_energy_queue_j=float(result.energy_virtual_queue_j.mean()))
    return trajectory, row, regional_delays
