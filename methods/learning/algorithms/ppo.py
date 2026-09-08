"""Device-aware PPO implementation with immutable rollout observations and masks."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PPOSettings:
    learning_rate: float = 3e-4
    value_learning_rate: float = 1e-3
    clip: float = .2
    epochs: int = 4
    minibatch_size: int = 256
    entropy: float = .001
    max_grad_norm: float = .5
    target_kl: float = .02


from methods.learning.buffers.rollout import Rollout

def gae_returns(rewards, values, dones, gamma=1., lam=.95):
    rewards, values = np.asarray(rewards), np.asarray(values)
    if rewards.shape != values.shape or len(dones) != len(rewards) or not dones[-1]:
        raise ValueError('GAE expects aligned complete episodes, ending at a true terminal')
    advantage = np.zeros_like(values[0], dtype=float)
    returns = np.empty_like(values, dtype=float)
    for step in reversed(range(len(rewards))):
        live = 1. - float(dones[step])
        next_value = values[step + 1] if step + 1 < len(values) else 0.
        delta = rewards[step] + gamma * next_value * live - values[step]
        advantage = delta + gamma * lam * live * advantage
        returns[step] = advantage + values[step]
    return returns.astype(np.float32)


class PPO:
    def __init__(self, actor, critic, settings=None):
        self.actor, self.critic = actor, critic
        self.settings = settings or PPOSettings()
        cfg = self.settings
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.value_learning_rate, eps=1e-5)

    def update(self, rollout):
        cfg = self.settings
        device = next(self.actor.parameters()).device
        obs = torch.as_tensor(rollout.observations, dtype=torch.float32, device=device)
        masks = torch.as_tensor(rollout.masks, dtype=torch.bool, device=device)
        actions = torch.as_tensor(rollout.actions, device=device)
        old_log_probs = torch.as_tensor(rollout.log_probs, dtype=torch.float32, device=device)
        state = torch.as_tensor(rollout.critic_observations, dtype=torch.float32, device=device)
        returns = torch.as_tensor(rollout.returns, dtype=torch.float32, device=device)
        advantages = returns - torch.as_tensor(rollout.values, dtype=torch.float32, device=device)
        if len(advantages) < 2 or not torch.isfinite(advantages).all():
            raise ValueError('PPO needs at least two finite samples')
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        reports = []
        stopped = False
        for _ in range(cfg.epochs):
            for indices in torch.randperm(len(actions), device=device).split(cfg.minibatch_size):
                log_probs, entropy = self.actor.evaluate(obs[indices], masks[indices], actions[indices])
                log_ratio = log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                kl = ((ratio - 1.) - log_ratio).mean()
                if not torch.isfinite(kl):
                    raise FloatingPointError('Nonfinite policy likelihood ratio')
                if kl.item() > 1.5 * cfg.target_kl:
                    stopped = True
                    break
                surrogate = torch.minimum(ratio * advantages[indices],
                                          ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantages[indices])
                actor_loss = -surrogate.mean() - cfg.entropy * entropy.mean()
                value_loss = F.mse_loss(self.critic(state[indices]), returns[indices])
                if not torch.isfinite(actor_loss + value_loss):
                    raise FloatingPointError('Nonfinite PPO loss')
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.critic_optimizer.step()
                reports.append([actor_loss.item(), value_loss.item(), kl.item(), entropy.mean().item()])
            if stopped:
                break
        if not reports:
            raise RuntimeError('Policy changed between collection and its first PPO update')
        return dict(zip(('actor_loss', 'value_loss', 'approx_kl', 'entropy'),
                        map(float, np.mean(reports, axis=0))), early_stop=float(stopped))
