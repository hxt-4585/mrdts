"""随机飞行与独立 Member 采样：ERS 排序，记录逐 UAV 的区域切换。"""

import numpy as np
import torch

from env.contracts import PlacementDecision
from env.types import EntityKind, EntityRef
from methods.random_uav_ppo.ers import ERS
from methods.random_uav_ppo.flight import RandomFlight
from methods.random_uav_ppo.support import slot_metrics
from methods.random_uav_ppo.ppo import Policy, Sample, finish_trajectory


class Adapter:
    def __init__(self, environment, flight_seed=46, resample_slots=10):
        self.env = environment
        self.members = environment.members.member_uav_count
        self.nodes = tuple(EntityRef(EntityKind.MEMBER_UAV, i) for i in range(self.members))
        self.member_obs_dim = 5 * self.members + 8
        self.flight = RandomFlight(environment, flight_seed, resample_slots)

    def policies(self, hidden, device):
        return torch.nn.ModuleList([
            Policy(self.member_obs_dim, self.member_obs_dim, self.members + 2, False, hidden)
            for _ in range(self.members)]).to(device)

    def member_inputs(self, keys):
        env = self.env
        energy_start, energy_stop = 4 + 4 * self.members, 4 + 5 * self.members
        observations = np.empty((len(keys), self.member_obs_dim), dtype=np.float32)
        for row, key in enumerate(keys):
            observation = env.observe_member(key)
            observations[row, :energy_start] = observation[:energy_start]
            observations[row, energy_start:] = observation[energy_stop:-1]
        regions = env.members.region_ids[:self.members]
        masks = np.ones((len(keys), self.members + 2), dtype=bool)
        masks[:, :self.members] = regions[None, :] == regions[[key.owner_member_id for key in keys], None]
        return observations, observations, masks

    def execution_node(self, action, key):
        if action < self.members:
            return self.nodes[action]
        if action == self.members:
            return EntityRef(EntityKind.BS, 0)
        return EntityRef(EntityKind.GROUND_DEVICE, key.user_id)


def decide(policy, inputs, deterministic):
    observations, states, masks = inputs
    device = next(policy.parameters()).device
    actions, logp, values = policy.sample(
        torch.as_tensor(observations, device=device), torch.as_tensor(states, device=device),
        torch.as_tensor(masks, device=device), deterministic)
    actions, logp, values = actions.cpu().numpy(), logp.cpu().numpy(), values.cpu().numpy()
    return [Sample(obs, state, mask, action, float(lp), float(value))
            for obs, state, mask, action, lp, value in zip(observations, states, masks, actions, logp, values)]


def collect_slot(adapter, members, *, deterministic=False, member_gamma=1., gae_lambda=.95,
                 member_selector=None, request_observer=None):
    env = adapter.env
    positions_before = env.members.positions[:adapter.members, :2].copy()
    before = env.members.region_ids[:adapter.members].copy()
    begin = env.begin_step(adapter.flight.actions(env, env.step_index))
    if request_observer is not None:
        request_observer(begin.requests)
    plan = ERS(env._runtime).plan(begin.requests)
    orders = {node.index: iter(order) for node, order in plan.member_orders.items()}
    trajectories = {owner: [] for owner in orders}
    pending = {}
    ready = begin.ready_member_ids
    while ready:
        keys = [next(orders[owner]) for owner in ready]
        inputs = adapter.member_inputs(keys)
        placements = {}
        for row, key in enumerate(keys):
            if member_selector is None:
                sample = decide(members[key.owner_member_id],
                                tuple(values[row:row + 1] for values in inputs), deterministic)[0]
            else:
                obs, state, mask = (values[row] for values in inputs)
                sample = Sample(obs, state, mask, member_selector(key, mask), 0., 0.)
            pending[key] = sample
            trajectories[key.owner_member_id].append(sample)
            placements[key] = PlacementDecision(adapter.execution_node(int(sample.action), key))
        ready = env.submit_schedule(placements)
        for key, reward in env.pop_member_rewards().items():
            pending.pop(key).reward = reward.compute_delay()
    result = env.end_step()
    for key, reward in result.member_rewards.items():
        pending.pop(key).reward = reward.compute_delay()
    samples = {owner: finish_trajectory(trajectory, 0., member_gamma, gae_lambda)
               for owner, trajectory in trajectories.items()}
    metrics = slot_metrics(result.slot_result, env.epoch_index, env.step_index - 1,
                           int(begin.flight_violations.sum()))
    # 环境仍提供全局时延奖励，仅用于业务指标，基线没有 Master 策略。
    metrics.update(global_delay_reward=result.master_reward,
        member_reward_sum=sum(s.reward for trajectory in samples.values() for s in trajectory),
        member_decisions=sum(map(len, samples.values())),
        total_energy_j=float(result.total_energy_j.sum()),
        mean_energy_queue_j=float(result.energy_virtual_queue_j.mean()),
        flight_displacement_mean_m=float(np.linalg.norm(
            env.members.positions[:adapter.members, :2] - positions_before, axis=1).mean()))
    regions = env.members.region_ids[:adapter.members]
    changed = regions != before
    away = regions != adapter.flight.regions
    metrics.update(flight_region_crossings=int(changed.sum()),
                   flight_away_members=int(away.sum()),
                   flight_empty_regions=int(np.count_nonzero(env.members.region_member_counts == 0)))
    away_samples = away_reward = 0
    for owner, position in enumerate(env.members.positions[:adapter.members]):
        decisions = samples.get(owner, ())
        reward = sum(s.reward for s in decisions)
        metrics.update({f'member_{owner}_x_m': float(position[0]),
                        f'member_{owner}_y_m': float(position[1]),
                        f'member_{owner}_region': int(regions[owner]),
                        f'member_{owner}_crossed': int(changed[owner]),
                        f'member_{owner}_away': int(away[owner]),
                        f'member_{owner}_decisions': len(decisions),
                        f'member_{owner}_reward_sum': reward})
        if away[owner]:
            away_samples += len(decisions)
            away_reward += reward
    metrics.update(away_member_decisions=away_samples, away_member_reward_sum=away_reward)
    return samples, metrics
