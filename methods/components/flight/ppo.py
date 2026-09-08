"""Continuous policy inference; the complete method supplies allowed observations."""

import torch


class PPOFlight:
    def __init__(self, actor=None):
        self.actor = actor

    @torch.no_grad()
    def decide(self, observations, masks, *, deterministic):
        distribution = self.actor.distribution(observations)
        latent = distribution.mean if deterministic else distribution.sample()
        log_probs, _ = self.actor.evaluate(observations, masks, latent)
        return self.actor.actions(latent, masks), latent, log_probs
