"""两类共享策略的 PPO；连续动作保存 tanh 前采样值以精确重算概率。

层内混合样本只用于 minibatch，GAE 必须先按各智能体的时间顺序计算。
Master 的联合 log-prob 只累加当前管理的 Member 的两个动作分量。
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from methods.hmappo.diagnostics import empty_stats, rollout_stats, final_stats, grad_norm


def mlp(inputs, outputs, hidden, gain):
    layers = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                           nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))
    for layer in (layers[0], layers[2], layers[4]):
        nn.init.orthogonal_(layer.weight, math.sqrt(2))
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, gain)
    return layers


class Policy(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, continuous, hidden=128):
        super().__init__()
        self.continuous = continuous
        self.actor = mlp(obs_dim, action_dim, hidden, .01)
        self.critic = mlp(state_dim, 1, hidden, 1.)
        if continuous:
            self.log_std = nn.Parameter(torch.full((action_dim,), -.5))

    def distribution(self, observations, masks):
        output = self.actor(observations)
        if self.continuous:
            return Normal(output, self.log_std.clamp(-5., 2.).exp())
        return Categorical(logits=output.masked_fill(~masks, -torch.inf))

    def score(self, observations, actions, masks):
        distribution = self.distribution(observations, masks)
        if self.continuous:
            jacobian = 2. * (math.log(2.) - actions - F.softplus(-2. * actions))
            logp = ((distribution.log_prob(actions) - jacobian) * masks).sum(-1)
            # tanh 分布熵没有闭式形式；用当前分布重参数采样估计，包含变换雅可比。
            latent = distribution.rsample()
            entropy = ((distribution.entropy() + 2. * (
                math.log(2.) - latent - F.softplus(-2. * latent))) * masks).sum(-1)
            return logp, entropy
        return distribution.log_prob(actions), distribution.entropy()

    @torch.no_grad()
    def sample(self, observations, states, masks, deterministic=False):
        distribution = self.distribution(observations, masks)
        if deterministic:
            actions = distribution.mean if self.continuous else distribution.probs.argmax(-1)
        else:
            actions = distribution.sample()
        if self.continuous:
            jacobian = 2. * (math.log(2.) - actions - F.softplus(-2. * actions))
            logp = ((distribution.log_prob(actions) - jacobian) * masks).sum(-1)
        else:
            logp = distribution.log_prob(actions)
        return actions, logp, self.critic(states).squeeze(-1)


def advantages(rewards, values, bootstrap, gamma, gae_lambda):
    """一个无内部终止点的轨迹片段；终止传 0，rollout 截断传下一状态价值。"""
    result = np.empty(len(rewards), dtype=np.float32)
    carry, next_value = 0., bootstrap
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        carry = delta + gamma * gae_lambda * carry
        result[index] = carry
        next_value = values[index]
    return result, result + np.asarray(values, dtype=np.float32)


@dataclass
class Sample:
    observation: np.ndarray
    state: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    reward: float = 0.
    advantage: float = 0.
    target: float = 0.


def finish_trajectory(samples, bootstrap, gamma, gae_lambda):
    adv, targets = advantages([s.reward for s in samples], [s.value for s in samples],
                              bootstrap, gamma, gae_lambda)
    for sample, advantage, target in zip(samples, adv, targets):
        sample.advantage, sample.target = float(advantage), float(target)
    return samples


def update(policy, optimizer, samples, *, ppo_epochs, minibatch_size, clip_ratio,
           entropy_coef, value_coef, max_grad_norm, diagnostics=True):
    """更新一套策略；Member 仅传入该 UAV 自身样本，无决策时允许空样本集。"""
    if not samples:
        return dict(samples=0, actor_change=0., policy_loss=None, value_loss=None, entropy=None, kl=None,
                    diagnostics_collected=False, **empty_stats(policy))
    device = next(policy.parameters()).device
    tensor = lambda values, dtype=torch.float32: torch.as_tensor(np.asarray(values), dtype=dtype, device=device)
    obs = tensor([s.observation for s in samples])
    state = tensor([s.state for s in samples])
    masks = tensor([s.mask for s in samples], torch.bool)
    actions = tensor([s.action for s in samples], torch.float32 if policy.continuous else torch.long)
    old_logp = tensor([s.log_prob for s in samples])
    targets = tensor([s.target for s in samples])
    adv = tensor([s.advantage for s in samples])
    active = masks.any(-1)
    stats = rollout_stats(policy, samples, actions, masks, targets, adv, active)
    # 空区域 Master 仍训练价值，但不参与 actor 损失或优势标准化。
    adv = (adv - adv[active].mean()) / (adv[active].std(unbiased=False) + 1e-8)
    actor_before = torch.cat([p.detach().flatten() for p in policy.actor.parameters()]).clone()
    log_std_before = policy.log_std.detach().clone() if policy.continuous else None
    totals = torch.zeros(4, device=device)
    grad_totals = torch.zeros(5, device=device)
    entropy_per_dim = torch.zeros((), device=device)
    actor_parameters = list(policy.actor.parameters())
    if policy.continuous:
        actor_parameters.append(policy.log_std)
    batches = 0
    for _ in range(ppo_epochs):
        for indices in torch.randperm(len(samples), device=device).split(minibatch_size):
            logp, entropy = policy.score(obs[indices], actions[indices], masks[indices])
            ratio = (logp - old_logp[indices]).exp()
            surrogate = torch.minimum(ratio * adv[indices],
                                      ratio.clamp(1. - clip_ratio, 1. + clip_ratio) * adv[indices])
            weights = active[indices].float()
            count = weights.sum().clamp_min(1.)
            policy_loss = -(surrogate * weights).sum() / count
            entropy_loss = (entropy * weights).sum() / count
            value_loss = .5 * (policy.critic(state[indices]).squeeze(-1) - targets[indices]).square().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if diagnostics:
                grad_totals[3] += grad_norm(actor_parameters)
                grad_totals[4] += grad_norm(policy.critic.parameters())
            norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            grad_totals[0] += norm
            grad_totals[1] += (norm > max_grad_norm).float()
            grad_totals[2] += (max_grad_norm / (norm + 1e-6)).clamp(max=1.)
            optimizer.step()
            with torch.no_grad():
                kl = ((ratio - 1. - (logp - old_logp[indices])) * weights).sum() / count
            totals += torch.stack((policy_loss, value_loss, entropy_loss, kl)).detach()
            if policy.continuous:
                entropy_per_dim += ((entropy.detach() / masks[indices].sum(-1).clamp_min(1))
                                    * weights).sum() / count
            batches += 1
    actor_after = torch.cat([p.detach().flatten() for p in policy.actor.parameters()])
    gradients = (grad_totals / batches).tolist()
    stats.update(grad_norm=gradients[0], grad_clip_fraction=gradients[1], grad_scale=gradients[2])
    if policy.continuous:
        stats.update(log_std_change=(policy.log_std.detach() - log_std_before).norm().item(),
                     entropy_per_dim=(entropy_per_dim / batches).item())
    if diagnostics:
        stats.update(actor_grad_norm=gradients[3], critic_grad_norm=gradients[4])
        stats.update(final_stats(policy, obs, state, actions, masks, old_logp, targets, active, clip_ratio))
    return dict(samples=len(samples), actor_change=(actor_after - actor_before).norm().item(),
                diagnostics_collected=diagnostics, **stats,
                **dict(zip(('policy_loss', 'value_loss', 'entropy', 'kl'), (totals / batches).tolist())))
