"""只在 method 侧适配 MRDTSEnv：固定两层策略采样，按 owner 保存异步轨迹。

低层优化本时隙内的环境 MemberReward，时隙结束视为低层终止；默认 gamma=1
表示按决策序列累计成本，不对不等长的事件间隔额外折扣。高层跨时隙学习。
Master 采用参数共享 CTDE，Member 按固定 ID 独立训练 PPO，仅使用局部观测。
不声称两层局部奖励严格等价于单一全局目标。
"""

import numpy as np
import torch

from env.contracts import PlacementDecision
from env.types import EntityKind, EntityRef
from methods.hmappo.ers import ERS
from methods.hmappo.support import slot_metrics
from methods.hmappo.ppo import Policy, Sample, finish_trajectory


class Adapter:
    def __init__(self, environment):
        self.env = environment
        self.members = environment.members.member_uav_count
        self.masters = environment.masters.num_uavs
        self.identities = np.eye(self.masters, dtype=np.float32)
        self.nodes = tuple(EntityRef(EntityKind.MEMBER_UAV, i) for i in range(self.members))
        self.global_dim = 2 * self.members + self.members * self.masters + 1
        self.master_obs_dim = 3 * self.members + self.masters
        self.member_obs_dim = 5 * self.members + 8
        self.master_state_dim = self.global_dim + self.masters
        self.member_state_dim = self.member_obs_dim

    def policies(self, hidden, device):
        return (Policy(self.master_obs_dim, self.master_state_dim, 2 * self.members, True, hidden).to(device),
                torch.nn.ModuleList([
                    Policy(self.member_obs_dim, self.member_state_dim, self.members + 2, False, hidden)
                    for _ in range(self.members)]).to(device))

    def global_state(self):
        env = self.env
        return np.concatenate((
            (env.members.positions[:self.members, :2] / env.region.config.side_length).ravel(),
            self.identities[env.members.region_ids[:self.members] - 1].ravel(),
            [env.step_index / env.env_config.steps_per_epoch],
        )).astype(np.float32)

    def master_inputs(self):
        env = self.env
        ownership = env.masters.region_ids[:, None] == env.members.region_ids[None, :self.members]
        # 环境每个 Member 返回 [x/L, y/L, Q/E]；时延策略只截取位置。
        observations = np.stack([np.concatenate((env.observe_master(i).reshape(self.members, 3)[:, :2].ravel(),
                                                ownership[i], self.identities[i]))
                                 for i in range(self.masters)]).astype(np.float32)
        states = np.concatenate((np.broadcast_to(self.global_state(), (self.masters, self.global_dim)),
                                 self.identities), axis=1)
        return observations, states, np.repeat(ownership, 2, axis=1)

    def member_inputs(self, keys):
        env = self.env
        # 跳过中间的 M 维能量队列，保留其后的计算等待时间与用户位置。
        energy_start, energy_stop = 4 + 4 * self.members, 4 + 5 * self.members
        observations = np.empty((len(keys), self.member_obs_dim), dtype=np.float32)
        for row, key in enumerate(keys):
            observation = env.observe_member(key)
            observations[row, :energy_start] = observation[:energy_start]
            observations[row, energy_start:] = observation[energy_stop:]
        regions = env.members.region_ids[:self.members]
        masks = np.ones((len(keys), self.members + 2), dtype=bool)
        masks[:, :self.members] = regions[None, :] == regions[[key.owner_member_id for key in keys], None]
        # 独立 PPO 的 critic 与 actor 使用同一份局部观测，无全局训练信息。
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


def collect_slot(adapter, master, members, *, deterministic=False, member_gamma=1., gae_lambda=.95,
                 boundary_penalty=.1, flight_actions=None, member_deterministic=None):
    env = adapter.env
    if flight_actions is None:
        master_samples = decide(master, adapter.master_inputs(), deterministic)
        actions = np.stack([np.tanh(sample.action) * sample.mask for sample in master_samples]).sum(0)
        ownership = np.stack([sample.mask[::2] for sample in master_samples])
    else:
        master_samples = []
        actions = flight_actions
        ownership = env.masters.region_ids[:, None] == env.members.region_ids[None, :adapter.members]
    if member_deterministic is None:
        member_deterministic = deterministic
    positions_before = env.members.positions[:adapter.members, :2].copy()
    begin = env.begin_step(actions.reshape(adapter.members, 2))
    displacement = np.linalg.norm(env.members.positions[:adapter.members, :2] - positions_before, axis=1)
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
            sample = decide(members[key.owner_member_id],
                            tuple(values[row:row + 1] for values in inputs), member_deterministic)[0]
            pending[key] = sample
            trajectories[key.owner_member_id].append(sample)
            placements[key] = PlacementDecision(adapter.execution_node(int(sample.action), key))
        ready = env.submit_schedule(placements)
        for key, reward in env.pop_member_rewards().items():
            pending.pop(key).reward = reward
    result = env.end_step()
    for key, reward in result.member_rewards.items():
        pending.pop(key).reward = reward
    delay_reward = result.master_reward
    penalties = boundary_penalty * np.count_nonzero(ownership & begin.flight_violations, axis=1)
    penalty_sum = float(penalties.sum())
    for sample, penalty in zip(master_samples, penalties):
        # 采样 mask 保存飞行前归属；每个 Member 的 x/y 动作共用一个归属标记。
        sample.reward = delay_reward - penalty
    member_samples = {owner: finish_trajectory(trajectory, 0., member_gamma, gae_lambda)
                      for owner, trajectory in trajectories.items()}
    metrics = slot_metrics(result.slot_result, env.epoch_index, env.step_index - 1,
                           int(begin.flight_violations.sum()))
    metrics.update(master_reward=delay_reward,
                   master_training_reward_mean=delay_reward - penalty_sum / len(ownership),
                   master_boundary_penalty_sum=penalty_sum,
                   member_reward_sum=sum(s.reward for samples in member_samples.values() for s in samples),
                   member_decisions=sum(map(len, member_samples.values())),
                   total_energy_j=float(result.total_energy_j.sum()),
                   mean_energy_queue_j=float(result.energy_virtual_queue_j.mean()))
    metrics['flight_displacement_mean_m'] = float(displacement.mean())
    for owner, owned in enumerate(ownership):
        metrics.update({f'master_{owner}_reward': delay_reward - penalties[owner],
                        f'master_{owner}_owned_members': int(owned.sum()),
                        f'master_{owner}_flight_violations': int(np.count_nonzero(owned & begin.flight_violations)),
                        f'master_{owner}_displacement_mean_m': float(displacement[owned].mean()) if owned.any() else None,
                        f'master_{owner}_boundary_penalty': penalties[owner]})
    return master_samples, member_samples, metrics


@torch.no_grad()
def master_bootstrap(adapter, policy):
    _, states, _ = adapter.master_inputs()
    device = next(policy.parameters()).device
    return policy.critic(torch.as_tensor(states, device=device)).squeeze(-1).cpu().numpy()
