"""Independent per-member PPO with slot team cost and local queue observations."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class MemberPolicy(nn.Module):
    def __init__(self, count, hidden=128):
        super().__init__()
        width = 6 * count + 8
        self.actor = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, count + 2))
        self.critic = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    @staticmethod
    def inputs(env, key):
        obs = env.observe_member(key).astype(np.float32)
        count = env.members.member_uav_count
        mask = np.ones(count + 2, bool)
        regions = env.members.region_ids[:count]
        mask[:count] = regions == regions[key.owner_member_id]
        return obs, mask

    def distribution(self, obs, mask):
        return Categorical(logits=self.actor(obs).masked_fill(~mask, -torch.inf))

    @torch.no_grad()
    def choose(self, obs, mask, deterministic):
        x, m = torch.from_numpy(obs), torch.from_numpy(mask)
        distribution = self.distribution(x, m)
        action = distribution.logits.argmax() if deterministic else distribution.sample()
        return int(action), float(distribution.log_prob(action)), float(self.critic(x).squeeze())


def update(policy, optimizer, records, epochs=4):
    if not records:
        return 0.
    obs = torch.tensor(np.stack([r[0] for r in records]))
    mask = torch.tensor(np.stack([r[1] for r in records]))
    actions = torch.tensor([r[2] for r in records])
    old = torch.tensor([r[3] for r in records])
    returns = torch.tensor([r[5] for r in records], dtype=torch.float32)
    advantage = returns - torch.tensor([r[4] for r in records])
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    losses = []
    for _ in range(epochs):
        for ids in torch.randperm(len(records)).split(256):
            distribution = policy.distribution(obs[ids], mask[ids])
            ratio = (distribution.log_prob(actions[ids]) - old[ids]).exp()
            surrogate = torch.minimum(ratio * advantage[ids], ratio.clamp(.8, 1.2) * advantage[ids])
            value = policy.critic(obs[ids]).flatten()
            loss = -surrogate.mean() + .5 * (value - returns[ids]).square().mean() - .01 * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))
