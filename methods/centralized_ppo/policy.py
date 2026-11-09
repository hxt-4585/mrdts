"""Terminal joint-action PPO with mirrored rollout sampling.

The action is one vector of deployment logits for the entire team. All action
dimensions share one likelihood ratio and one terminal system return. Fixed
exploration schedules and exact Gaussian KL backtracking control policy drift.
PPO: https://arxiv.org/abs/1707.06347
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


class JointPPO(nn.Module):
    def __init__(self, observation, initial, hidden, learning_rate):
        super().__init__()
        self.shape = initial.shape
        self.register_buffer('observation', torch.as_tensor(observation, dtype=torch.float32))
        initial_tensor = torch.as_tensor(initial, dtype=torch.float32).flatten()
        # The finite logistic parameterization is the action domain itself.
        self.register_buffer('origin', torch.logit(initial_tensor.clamp(.001, .999)))
        self.actor = nn.Sequential(nn.Linear(len(observation), hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, initial.size))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

    def mean(self):
        return self.origin + self.actor(self.observation)

    def log_prob(self, latent, std):
        # The sigmoid Jacobian cancels in the new/old ratio for fixed actions.
        return Normal(self.mean(), std).log_prob(latent).sum(-1)

    @torch.no_grad()
    def sample(self, pairs, std):
        mean = self.mean().detach().clone()
        noise = torch.randn(pairs, len(mean)) * std
        return torch.cat((mean + noise, mean - noise)), mean

    @torch.no_grad()
    def targets(self):
        return self.mean().sigmoid().reshape(self.shape).numpy().copy()

    def update(self, actions, rewards, old_mean, std, *, epochs, clip_ratio, target_kl):
        old_distribution = Normal(old_mean, std)
        old_logp = old_distribution.log_prob(actions).sum(-1)
        pairs = len(rewards) // 2
        # Excluding both members of a mirrored pair gives an action-independent
        # baseline. Each marginal action still has the correct Gaussian law.
        pair_sums = rewards[:pairs] + rewards[pairs:]
        baselines = (rewards.sum() - pair_sums) / (len(rewards) - 2)
        advantage = rewards - baselines.repeat(2)
        advantage = advantage / (advantage.std(unbiased=False) + 1e-8)
        parameters = list(self.actor.parameters())
        accepted = 0
        for _ in range(epochs):
            ratio = (self.log_prob(actions, std) - old_logp).exp()
            surrogate = torch.minimum(ratio * advantage,
                                      ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
            loss = -surrogate.mean()
            before = [p.detach().clone() for p in parameters]
            optimizer_before = copy.deepcopy(self.optimizer.state_dict())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.)
            self.optimizer.step()
            after = [p.detach().clone() for p in parameters]
            with torch.no_grad():
                # Bound total drift from the rollout policy, not just each step.
                for attempt in range(12):
                    fraction = .5 ** attempt
                    for parameter, start, end in zip(parameters, before, after):
                        parameter.copy_(start + fraction * (end - start))
                    kl = kl_divergence(old_distribution, Normal(self.mean(), std)).sum()
                    if kl <= target_kl:
                        break
                else:
                    for parameter, start in zip(parameters, before):
                        parameter.copy_(start)
                    self.optimizer.load_state_dict(optimizer_before)
                    break
            accepted += 1
        with torch.no_grad():
            kl = kl_divergence(old_distribution, Normal(self.mean(), std)).sum()
        return dict(kl=float(kl), policy_loss=float(loss.detach()), accepted_steps=accepted,
                    reward_std=float(rewards.std(unbiased=False)))
