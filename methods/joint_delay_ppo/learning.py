"""Independent local categorical actors and value functions for task placement."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class LocalPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        def network(output, output_gain):
            layers = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output))
            for index in (0, 2, 4):
                nn.init.orthogonal_(layers[index].weight, output_gain if index == 4 else math.sqrt(2.))
                nn.init.zeros_(layers[index].bias)
            return layers
        self.actor = network(action_dim, .01)
        self.critic = network(1, 1.)

    def distribution(self, observations, masks):
        return Categorical(logits=self.actor(observations).masked_fill(~masks, -torch.inf))

    @torch.no_grad()
    def act(self, observation, mask, deterministic=False):
        device = next(self.parameters()).device
        obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
        valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
        distribution = self.distribution(obs, valid)
        action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(self.critic(obs).item())


def make_model(member_count, obs_dim, config, device):
    return nn.ModuleList([LocalPolicy(obs_dim, member_count + 2, config['hidden'])
                          for _ in range(member_count)]).to(device)


def make_optimizers(model, config):
    return [torch.optim.Adam(policy.parameters(), lr=config['learning_rate'], eps=1e-5) for policy in model]


@dataclass
class Decision:
    observation: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float = 0.
    advantage: float = 0.
    target: float = 0.
    key: object = None


def finish_trajectory(samples, gamma, gae_lambda):
    """The completed slot terminates each member's own decision sequence."""
    future_value = accumulated = 0.
    for sample in reversed(samples):
        residual = sample.reward + gamma * future_value - sample.value
        accumulated = residual + gamma * gae_lambda * accumulated
        sample.advantage = accumulated
        sample.target = accumulated + sample.value
        future_value = sample.value


def update(model, optimizers, trajectories, config):
    totals = np.zeros(4, dtype=np.float64)
    weighted_count = sample_count = 0
    for member, (policy, optimizer) in enumerate(zip(model, optimizers)):
        samples = [sample for trajectory in trajectories for sample in trajectory[member]]
        size = len(samples)
        sample_count += size
        if not size:
            continue
        device = next(policy.parameters()).device
        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)
        observations = tensor([sample.observation for sample in samples])
        masks = tensor([sample.mask for sample in samples], torch.bool)
        actions = tensor([sample.action for sample in samples], torch.long)
        previous_logp = tensor([sample.log_prob for sample in samples])
        targets = tensor([sample.target for sample in samples])
        advantages = tensor([sample.advantage for sample in samples])
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        for _ in range(config['ppo_epochs']):
            for indices in torch.randperm(size, device=device).split(config['minibatch_size']):
                distribution = policy.distribution(observations[indices], masks[indices])
                logp = distribution.log_prob(actions[indices])
                difference = logp - previous_logp[indices]
                ratio = difference.exp()
                clipped = ratio.clamp(1. - config['clip_ratio'], 1. + config['clip_ratio'])
                policy_loss = -torch.minimum(ratio * advantages[indices],
                                              clipped * advantages[indices]).mean()
                value = policy.critic(observations[indices]).squeeze(-1)
                value_loss = .5 * (value - targets[indices]).square().mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + config['value_coef'] * value_loss - config['entropy_coef'] * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config['max_grad_norm'])
                optimizer.step()
                kl = (ratio.detach() - 1. - difference.detach()).mean()
                weight = indices.numel()
                totals += np.asarray([policy_loss.item(), value_loss.item(), entropy.item(), kl.item()]) * weight
                weighted_count += weight
    if weighted_count:
        totals /= weighted_count
    return dict(samples=sample_count, **dict(zip(('policy_loss', 'value_loss', 'entropy', 'kl'), totals.tolist())))
