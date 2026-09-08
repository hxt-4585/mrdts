"""Two shared actor types; no global critic features enter either actor."""

import torch
from torch import nn
from torch.distributions import Categorical, Normal


from methods.learning.networks.mlp import mlp

class MemberActor(nn.Module):
    def __init__(self, feature_count, hidden=64):
        super().__init__()
        self.scorer = mlp(feature_count, 1, hidden, output_gain=.01)
        # A learnable local prior avoids a near-constant timeout reward at cold
        # start under the default 1000-task workload. No candidate is removed.
        self.ground_bias = nn.Parameter(torch.tensor(4.))

    def distribution(self, observations, mask):
        if not mask.any(dim=-1).all():
            raise ValueError('Every active Member needs at least one legal action')
        logits = self.scorer(observations).squeeze(-1) + self.ground_bias * observations[..., 11]
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))

    def evaluate(self, observations, mask, actions):
        dist = self.distribution(observations, mask)
        return dist.log_prob(actions), dist.entropy()


class MasterActor(nn.Module):
    """Gaussian latent action followed by tanh; PPO uses latent log likelihood.

    For a fixed stored mask, tanh's Jacobian cancels exactly in the PPO ratio.
    Entropy regularization below is latent Gaussian entropy, not tanh entropy.
    """

    def __init__(self, observation_count, action_count, hidden=64):
        super().__init__()
        self.mean = mlp(observation_count, action_count, hidden, output_gain=.01)
        self.log_std = nn.Parameter(torch.full((action_count,), -.7))

    def distribution(self, observations):
        return Normal(self.mean(observations), self.log_std.clamp(-3., .5).exp())

    @staticmethod
    def actions(latent, mask):
        return torch.tanh(latent) * mask

    def evaluate(self, observations, mask, latent):
        dist = self.distribution(observations)
        return (dist.log_prob(latent) * mask).sum(-1), (dist.entropy() * mask).sum(-1)


