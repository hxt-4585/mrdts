"""Masked candidate inference, independently batched across active Members."""

import torch


class PPOScheduling:
    def __init__(self, actor=None):
        self.actor = actor

    @torch.no_grad()
    def decide(self, observations, masks, *, deterministic):
        distribution = self.actor.distribution(observations, masks)
        actions = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions)
