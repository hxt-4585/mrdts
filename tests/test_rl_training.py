"""Policy probability, update and training lifecycle checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Install requirements-rl.txt for RL tests')
class TestRLTraining(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('methods.rl_baseline.networks'),
                             'The optional policy implementation must exist')
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_member_mask_gives_illegal_actions_zero_probability(self):
        import torch
        from methods.rl_baseline.networks import MemberActor
        actor = MemberActor(23, 32)
        features = torch.randn(3, 14, 23)
        mask = torch.zeros(3, 14, dtype=torch.bool)
        mask[:, [0, 2, 13]] = True
        distribution = actor.distribution(features, mask)
        self.assertTrue(torch.equal(distribution.probs[~mask], torch.zeros(33)))
        for _ in range(20):
            actions = distribution.sample()
            self.assertTrue(mask.gather(1, actions[:, None]).all())
        with self.assertRaises(ValueError):
            actor.distribution(features, torch.zeros_like(mask))

    def test_master_excludes_unowned_components_from_log_prob(self):
        import torch
        from methods.rl_baseline.networks import MasterActor
        actor = MasterActor(10, 6, 32)
        observations = torch.zeros(2, 10)
        mask = torch.tensor([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=torch.bool)
        latent = torch.zeros(2, 6)
        original, _ = actor.evaluate(observations, mask, latent)
        latent[~mask] = 99.
        changed, _ = actor.evaluate(observations, mask, latent)
        torch.testing.assert_close(original, changed)
        self.assertTrue(torch.equal(actor.actions(latent, mask)[~mask], torch.zeros(8)))

    def test_local_initialization_retains_trainable_offloading_support(self):
        import torch
        from methods.rl_baseline.networks import MemberActor
        actor = MemberActor(23, 32)
        features = torch.zeros(1, 14, 23)
        features[0, 0, 11] = 1.  # Own-ground candidate kind feature.
        mask = torch.ones(1, 14, dtype=torch.bool)
        probabilities = actor.distribution(features, mask).probs
        self.assertGreater(float(probabilities[0, 0].detach()), .8,
                           'Default initialization should avoid almost-all-timeout exploration')
        self.assertTrue((probabilities > 0).all(), 'Every legal candidate remains available')
        (-actor.distribution(features, mask).log_prob(torch.tensor([1]))).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters()))

    def test_gae_stops_at_episode_boundary(self):
        from methods.rl_baseline.ppo import gae_returns
        rewards = np.array([[-1.], [-2.], [-8.]])
        values = np.zeros_like(rewards)
        returns = gae_returns(rewards, values, [False, True, True], gamma=1., lam=1.)
        np.testing.assert_allclose(returns[:, 0], [-3., -2., -8.])

    def test_real_rollout_update_freeze_and_checkpoint_roundtrip(self):
        import torch
        from methods.rl_baseline.adapter import DelayEnvironment
        from methods.rl_baseline.runner import Learner, collect_member, collect_master
        from methods.rl_baseline.ppo import PPOSettings
        env = DelayEnvironment()
        learner = Learner(env, hidden=32, settings=PPOSettings(epochs=1))
        samples, metrics = collect_member(env, learner, slots=2)
        self.assertEqual(len(samples.actions), 2000)
        self.assertEqual(len(metrics), 2)
        before = [p.detach().clone() for p in learner.member.parameters()]
        losses = learner.member_ppo.update(samples)
        self.assertTrue(all(np.isfinite(v) for v in losses.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, learner.member.parameters())))
        before = [p.detach().clone() for p in learner.member.parameters()]
        master_samples, metrics = collect_master(env, learner, slots=2)
        self.assertEqual(len(master_samples.actions), 8)
        learner.master_ppo.update(master_samples)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, learner.member.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            learner.save(path, {'member_updates': 1})
            other = Learner(env, hidden=32, settings=PPOSettings(epochs=1))
            metadata = other.load(path)
            self.assertEqual(metadata['member_updates'], 1)
            for a, b in zip(learner.member.parameters(), other.member.parameters()):
                torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
