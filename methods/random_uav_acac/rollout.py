"""Event-driven collection. Histories reset at the scheduling-slot terminal.

Each joint decision event advances only ready agents. Rewards are attributed to
their actual owner and completion time, including completions after the last
decision. Time is continuous scheduling time divided by nominal slot duration.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.distributions import Categorical

from env.contracts import PlacementDecision
from env.types import EntityKind, EntityRef
from methods.random_uav_acac.flight import RandomFlight
from methods.random_uav_acac.scheduling import order_requests


@dataclass
class Event:
    time: float
    owners: tuple
    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    completions: list = field(default_factory=list)


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


class Adapter:
    def __init__(self, env, seed=45, resample_slots=10):
        self.env = env
        self.count = env.members.member_uav_count
        self.obs_dim = 5 * self.count + 8
        self.flight = RandomFlight(env, seed, resample_slots)
        self.nodes = tuple(EntityRef(EntityKind.MEMBER_UAV, i) for i in range(self.count))

    def inputs(self, keys):
        observations = np.empty((len(keys), self.obs_dim), np.float32)
        queue_start, queue_stop = 4 + 4 * self.count, 4 + 5 * self.count
        for index, key in enumerate(keys):
            full = self.env.observe_member(key)
            observations[index, :queue_start] = full[:queue_start]
            observations[index, queue_start:] = full[queue_stop:-1]
        regions = self.env.members.region_ids[:self.count]
        masks = np.ones((len(keys), self.count + 2), bool)
        masks[:, :self.count] = regions[None, :] == regions[[key.owner_member_id for key in keys], None]
        return observations, masks

    def node(self, key, action):
        if action < self.count:
            return self.nodes[action]
        if action == self.count:
            return EntityRef(EntityKind.BS, 0)
        return EntityRef(EntityKind.GROUND_DEVICE, key.user_id)


@torch.no_grad()
def collect_slot(adapter, model, *, deterministic=False, member_gamma=1., gae_lambda=.95,
                 selector=None, request_observer=None):
    from methods.random_uav_acac.learning import advantages

    env, count = adapter.env, adapter.count
    before = env.members.region_ids[:count].copy()
    begin = env.begin_step(adapter.flight.actions(env, env.step_index))
    runtime = env._runtime
    origin, duration = runtime.slot_start, env.env_config.scheduling_duration_s
    if request_observer is not None:
        request_observer(begin.requests)
    orders = {owner: iter(keys) for owner, keys in order_requests(runtime, begin.requests).items()}
    state = model.initial_state() if model is not None else None
    device = next(model.parameters()).device if model is not None else torch.device('cpu')
    events, ready = [], begin.ready_member_ids
    owner_rewards = np.zeros(count)
    while ready:
        keys = [next(orders[owner]) for owner in ready]
        observations, masks = adapter.inputs(keys)
        time = (runtime.now - origin) / duration
        if selector is None:
            inputs = {owner: torch.as_tensor(obs, device=device) for owner, obs in zip(ready, observations)}
            logits, values, state = model.advance(inputs, time, state)
            actions, logps = [], []
            for owner, mask in zip(ready, masks):
                distribution = Categorical(logits=logits[owner].masked_fill(
                    ~torch.as_tensor(mask, device=device), -torch.inf))
                action = distribution.probs.argmax() if deterministic else distribution.sample()
                actions.append(int(action))
                logps.append(float(distribution.log_prob(action)))
            values = values.cpu().numpy()
        else:
            actions = [selector(key, obs, mask) for key, obs, mask in zip(keys, observations, masks)]
            logps, values = np.zeros(len(keys)), np.zeros(count)
        event = Event(time, tuple(ready), observations, masks, np.array(actions), np.array(logps), values)
        events.append(event)
        ready = env.submit_schedule({key: PlacementDecision(adapter.node(key, action))
                                     for key, action in zip(keys, actions)})
        for key, reward in env.pop_member_rewards().items():
            completed = (runtime.tasks[key].compute_finish_at - origin) / duration
            delay_reward = reward.compute_delay()
            event.completions.append((key.owner_member_id, completed, delay_reward))
            owner_rewards[key.owner_member_id] += delay_reward
    result = env.end_step()
    for key, reward in result.member_rewards.items():
        delay_reward = reward.compute_delay()
        events[-1].completions.append((key.owner_member_id,
                                      (runtime.tasks[key].compute_finish_at - origin) / duration, delay_reward))
        owner_rewards[key.owner_member_id] += delay_reward
    times = np.array([e.time for e in events] + [(result.slot_result.slot_end - origin) / duration])
    values = np.array([e.values for e in events], dtype=np.float32).reshape(-1, count)
    rewards = np.zeros_like(values)
    active = np.zeros(count, bool)
    for index, event in enumerate(events):
        active[list(event.owners)] = True
        for owner, completion, reward in event.completions:
            rewards[index, owner] += member_gamma ** (completion - event.time) * reward
    adv, targets = advantages(rewards, values, times, member_gamma, gae_lambda)
    trajectory = Trajectory(events, rewards, values, times, adv, targets, active)
    dags = result.slot_result.dags
    delay = sum(d.completion_time - result.slot_result.slot_start for d in dags)
    regions = env.members.region_ids[:count]
    metrics = dict(episode=env.epoch_index, slot=env.step_index - 1, dag_count=len(dags),
                   failed_dags=sum(d.failed for d in dags), delay_sum_s=delay,
                   mean_delay_s=delay / len(dags), global_delay_reward=result.master_reward,
                   member_decisions=trajectory.samples, member_reward_sum=float(owner_rewards.sum()),
                   joint_decision_events=len(events), total_energy_j=float(result.total_energy_j.sum()),
                   flight_violations=int(begin.flight_violations.sum()),
                   flight_region_crossings=int(np.count_nonzero(regions != before)),
                   flight_away_members=int(np.count_nonzero(regions != adapter.flight.initial_regions)),
                   flight_empty_regions=int(np.count_nonzero(env.members.region_member_counts == 0)))
    for owner in range(count):
        metrics[f'member_{owner}_decisions'] = sum(owner in event.owners for event in events)
        metrics[f'member_{owner}_reward_sum'] = float(owner_rewards[owner])
    return trajectory, metrics
