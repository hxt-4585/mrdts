"""Independent per-member clipped PPO with slot-terminal GAE."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def member_input(raw, count):
    # Omit energy queues, keep waiting times, user coordinates and unfinished-DAG share.
    return np.concatenate((raw[:4 + 4 * count], raw[4 + 5 * count:])).astype(np.float32)


@dataclass
class Transition:
    observation: np.ndarray
    mask: np.ndarray
    action: int
    log_probability: float
    value: float
    reward: float = 0.
    advantage: float = 0.
    target: float = 0.


def advantages(sequence, gamma, trace):
    following_value = running = 0.
    for sample in reversed(sequence):
        residual = sample.reward + gamma * following_value - sample.value
        running = residual + gamma * trace * running
        sample.advantage = running
        sample.target = running + sample.value
        following_value = sample.value


class MemberPolicy(nn.Module):
    def __init__(self, inputs, actions, hidden):
        super().__init__()
        def mlp(outputs):
            return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, outputs))
        self.actor = mlp(actions)
        self.critic = mlp(1)

    def distribution(self, x, mask):
        return Categorical(logits=self.actor(x).masked_fill(~mask, -torch.inf))


class PPO:
    def __init__(self, count, config, device=None):
        self.config = config
        self.device = torch.device(device or config['device'])
        self.policies = nn.ModuleList([MemberPolicy(5 * count + 9, count + 2, config['hidden'])
                                       for _ in range(count)]).to(self.device)
        self.optimizers = [torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])
                           for policy in self.policies]

    @torch.no_grad()
    def act(self, member, observation, mask, deterministic=False):
        policy = self.policies[member]
        x = torch.as_tensor(observation, device=self.device)
        valid = torch.as_tensor(mask, device=self.device)
        distribution = policy.distribution(x, valid)
        action = distribution.logits.argmax() if deterministic else distribution.sample()
        return Transition(observation, mask, int(action.item()), float(distribution.log_prob(action).item()),
                          float(policy.critic(x).item()))

    def learn(self, batches):
        cfg, results = self.config, []
        for member, (policy, optimizer) in enumerate(zip(self.policies, self.optimizers)):
            sequence = [sample for batch in batches for sample in batch[member]]
            if not sequence:
                continue
            def tensor(values, dtype=torch.float32):
                return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)
            x = tensor([s.observation for s in sequence])
            masks = tensor([s.mask for s in sequence], torch.bool)
            actions = tensor([s.action for s in sequence], torch.long)
            old_logp = tensor([s.log_probability for s in sequence])
            targets = tensor([s.target for s in sequence])
            adv = tensor([s.advantage for s in sequence])
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            for _ in range(cfg['ppo_epochs']):
                order = torch.randperm(len(sequence), device=self.device)
                for indices in order.split(cfg['minibatch_size']):
                    distribution = policy.distribution(x[indices], masks[indices])
                    ratio = (distribution.log_prob(actions[indices]) - old_logp[indices]).exp()
                    clipped = ratio.clamp(1 - cfg['clip_ratio'], 1 + cfg['clip_ratio'])
                    actor_loss = -torch.minimum(ratio * adv[indices], clipped * adv[indices]).mean()
                    value_loss = (policy.critic(x[indices]).flatten() - targets[indices]).square().mean()
                    loss = actor_loss + cfg['value_coef'] * value_loss - cfg['entropy_coef'] * distribution.entropy().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg['max_grad_norm'])
                    optimizer.step()
                    results.append(loss.detach().item())
        return float(np.mean(results)) if results else 0.

    def weights(self):
        return {name: value.detach().cpu().clone() for name, value in self.policies.state_dict().items()}

    def state(self):
        return dict(weights=self.weights(), optimizers=[optimizer.state_dict() for optimizer in self.optimizers])

    def restore(self, state):
        self.policies.load_state_dict(state['weights'])
        for optimizer, saved in zip(self.optimizers, state['optimizers']):
            optimizer.load_state_dict(saved)
