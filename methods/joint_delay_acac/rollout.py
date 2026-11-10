"""One scheduling slot with ERS ordering and completion-time reward buckets."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.distributions import Categorical

from env.contracts import PlacementDecision
from env.runtime.dag_runtime import topological_order
from env.types import EntityKind, EntityRef, TaskKey
from .learning import advantages


@dataclass
class Event:
    time: float
    owners: tuple
    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray


@dataclass
class Trajectory:
    events: list
    rewards: np.ndarray
    values: np.ndarray
    times: np.ndarray
    advantages: np.ndarray
    targets: np.ndarray
    active: np.ndarray

    @property
    def samples(self):
        return sum(len(event.owners) for event in self.events)


def order_requests(runtime, requests):
    """Core-weighted average computation and routed communication upward ranks."""
    contexts, transfer_cost = {}, {}
    priorities = {}
    for request in requests:
        if request.owner_member is None:
            continue
        owner, user = request.owner_member, request.ground_device
        context = owner, user
        if context not in contexts:
            nodes = runtime.candidate_execution_nodes(owner, user)
            frequencies = {node: runtime.servers[node].core_frequencies for node in nodes}
            total_cores = sum(len(frequencies[node]) for node in nodes)
            computation = math.fsum(1. / f for fs in frequencies.values() for f in fs) / total_cores
            routes = []
            for source in nodes:
                for target in nodes:
                    route = runtime.route_planner.predecessor_route(source, target, user, owner)
                    for hop in route.hops:
                        if hop not in transfer_cost:
                            transfer_cost[hop] = runtime.transfer_duration_s(hop, 1.)
                    routes.append(len(frequencies[source]) * len(frequencies[target]) *
                                  math.fsum(transfer_cost[hop] for hop in route.hops))
            contexts[context] = computation, math.fsum(routes) / total_cores**2
        computation, communication = contexts[context]
        dag = request.dag
        topology = topological_order(dag)
        children = {node: [] for node in topology}
        for source, target in dag.edges:
            children[source].append(target)
        rank = {}
        for node in reversed(topology):
            rank[node] = float(dag.node_features[node, 1]) * computation + max(
                (rank[child] + float(dag.edge_features[node, child]) * 1000. * communication
                 for child in children[node]), default=0.)
        for position, node in enumerate(topology):
            priorities[TaskKey(*request.key, node)] = rank[node], position
    sorted_keys = sorted(priorities, key=lambda key: (-priorities[key][0], key.owner_member_id,
                                                    key.user_id, key.dag_id, priorities[key][1]))
    orders = {}
    for key in sorted_keys:
        orders.setdefault(key.owner_member_id, []).append(key)
    return orders


def member_inputs(env, keys):
    count = env.members.member_uav_count
    observations = np.empty((len(keys), 5 * count + 8), dtype=np.float32)
    start, stop = 4 + 4 * count, 4 + 5 * count
    masks = np.ones((len(keys), count + 2), dtype=bool)
    regions = env.members.region_ids[:count]
    for row, key in enumerate(keys):
        full = env.observe_member(key)
        observations[row, :start] = full[:start]
        observations[row, start:] = full[stop:-1]
        masks[row, :count] = regions == regions[key.owner_member_id]
    return observations, masks


def bucket_rewards(event_times, count, completions, gamma):
    rewards = np.zeros((len(event_times), count), dtype=np.float32)
    totals = np.zeros(count)
    for owner, completed, reward, available_event in completions:
        # Availability resolves same-time causal ordering: a completion cannot
        # be attributed to an event that had not been observed when it arrived.
        bucket = min(int(np.searchsorted(event_times, completed, side='right')) - 1, available_event)
        rewards[bucket, owner] += gamma ** (completed - event_times[bucket]) * reward
        totals[owner] += reward
    return rewards, totals


@torch.no_grad()
def collect_slot(env, model, actions, config, deterministic=False):
    count = env.members.member_uav_count
    begin = env.begin_step(actions)
    runtime = env._runtime
    origin, duration = runtime.slot_start, env.env_config.scheduling_duration_s
    orders = {owner: iter(keys) for owner, keys in order_requests(runtime, begin.requests).items()}
    events, completions = [], []
    state = model.initial_state()
    ready = begin.ready_member_ids
    device = next(model.parameters()).device
    member_nodes = [EntityRef(EntityKind.MEMBER_UAV, owner) for owner in range(count)]
    bs = EntityRef(EntityKind.BS, 0)
    while ready:
        keys = [next(orders[owner]) for owner in ready]
        observations, masks = member_inputs(env, keys)
        time = (runtime.now - origin) / duration
        logits, values, state = model.advance(
            {owner: torch.as_tensor(obs, device=device) for owner, obs in zip(ready, observations)}, time, state)
        selected, logps = [], []
        placements = {}
        for owner, key, mask in zip(ready, keys, masks):
            distribution = Categorical(logits=logits[owner].masked_fill(~torch.as_tensor(mask, device=device), -torch.inf))
            action = distribution.probs.argmax() if deterministic else distribution.sample()
            choice = int(action)
            selected.append(choice)
            logps.append(float(distribution.log_prob(action)))
            node = member_nodes[choice] if choice < count else bs if choice == count else EntityRef(EntityKind.GROUND_DEVICE, key.user_id)
            placements[key] = PlacementDecision(node)
        events.append(Event(time, tuple(ready), observations, masks, np.asarray(selected),
                            np.asarray(logps, dtype=np.float32), values.cpu().numpy()))
        ready = env.submit_schedule(placements)
        for key, reward in env.pop_member_rewards().items():
            completions.append((key.owner_member_id, (runtime.tasks[key].compute_finish_at - origin) / duration,
                                reward.compute_delay(), len(events) - 1))
    result = env.end_step()
    for key, reward in result.member_rewards.items():
        completions.append((key.owner_member_id, (runtime.tasks[key].compute_finish_at - origin) / duration,
                            reward.compute_delay(), len(events) - 1))
    times = np.asarray([event.time for event in events] + [(result.slot_result.slot_end - origin) / duration])
    values = np.asarray([event.values for event in events], dtype=np.float32).reshape(-1, count)
    rewards, owner_rewards = bucket_rewards(times[:-1], count, completions, config['member_gamma'])
    active = np.zeros(count, dtype=bool)
    for event in events:
        active[list(event.owners)] = True
    adv, targets = advantages(rewards, values, times, config['member_gamma'], config['gae_lambda'])
    trajectory = Trajectory(events, rewards, values, times, adv, targets, active)
    slot = result.slot_result
    regional = np.zeros(env.region.config.region_count)
    for dag in slot.dags:
        regional[int(env.users.region_ids[dag.key[1]]) - 1] += dag.completion_time - slot.slot_start
    row = dict(delay_sum_s=float(regional.sum()), dag_count=len(slot.dags),
               failed_dags=sum(dag.failed for dag in slot.dags),
               flight_violations=int(begin.flight_violations.sum()), member_decisions=trajectory.samples,
               member_reward_sum=float(owner_rewards.sum()), joint_decision_events=len(events),
               total_energy_j=float(result.total_energy_j.sum()), global_delay_reward=result.master_reward,
               episode=env.epoch_index, slot=env.step_index - 1)
    row['mean_delay_s'] = row['delay_sum_s'] / len(slot.dags)
    return trajectory, row, regional
