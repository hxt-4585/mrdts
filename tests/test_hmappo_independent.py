import csv
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from methods.hmappo.ppo import update
from methods.hmappo.rollout import Adapter, collect_slot
from methods.hmappo.run import train, load_checkpoint, evaluate
from methods.hmappo.support import build_scene, resolved_settings


class TestIndependentMembers(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_rollout_routes_by_owner_and_update_is_isolated(self):
        env = build_scene(resolved_settings(users=4, dag_nodes=3), episode=0)
        adapter = Adapter(env)
        master, members = adapter.policies(16, 'cpu')
        self.assertEqual(len(members), adapter.members)
        parameter_ids = [{id(p) for p in policy.parameters()} for policy in members]
        for i in range(len(members)):
            for j in range(i):
                self.assertTrue(parameter_ids[i].isdisjoint(parameter_ids[j]))
        _, trajectories, metrics = collect_slot(adapter, master, members, deterministic=True)
        self.assertEqual(sum(map(len, trajectories.values())), metrics['member_decisions'])
        for owner, samples in trajectories.items():
            for sample in samples:
                self.assertEqual(np.argmax(sample.observation[4 + adapter.members:4 + 2 * adapter.members]), owner)
                np.testing.assert_array_equal(sample.state, sample.observation)
                with torch.no_grad():
                    logp, _ = members[owner].score(
                        torch.as_tensor(sample.observation[None]), torch.as_tensor([sample.action]),
                        torch.as_tensor(sample.mask[None]))
                self.assertAlmostEqual(logp.item(), sample.log_prob, places=6)
        before = [{k: v.clone() for k, v in policy.state_dict().items()} for policy in members]
        owner = next(owner for owner, samples in trajectories.items() if samples)
        optimizer = torch.optim.Adam(members[owner].parameters(), lr=.001)
        update(members[owner], optimizer, trajectories[owner], ppo_epochs=1, minibatch_size=64,
               clip_ratio=.2, entropy_coef=.01, value_coef=.5, max_grad_norm=.5)
        self.assertTrue(any(not torch.equal(value, before[owner][key])
                            for key, value in members[owner].state_dict().items()))
        for i, policy in enumerate(members):
            if i != owner:
                for key, value in policy.state_dict().items():
                    self.assertTrue(torch.equal(value, before[i][key]))

    def test_training_updates_owners_separately_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix='mrdts_ippo_') as output:
            calls = []

            def tracked_update(policy, optimizer, samples, **options):
                calls.append((policy, optimizer, samples))
                return update(policy, optimizer, samples, **options)

            with patch('methods.hmappo.run.update', side_effect=tracked_update):
                path = train(episodes=1, slots=2, users=4, dag_nodes=3, hidden=16,
                             rollout_slots=2, ppo_epochs=1, output_root=output)
            env, master, members, checkpoint = load_checkpoint(path / 'checkpoint.pt')
            self.assertEqual(len(calls), len(members) + 1)
            self.assertEqual(len({id(optimizer) for _, optimizer, _ in calls}), len(calls))
            count = env.members.member_uav_count
            self.assertEqual(len(checkpoint['member_optimizers']), count)
            for owner, (policy, _, samples) in enumerate(calls[1:]):
                for sample in samples:
                    self.assertEqual(np.argmax(sample.observation[4 + count:4 + 2 * count]), owner)
                for key, value in policy.state_dict().items():
                    self.assertTrue(torch.equal(value, members[owner].state_dict()[key]))
            self.assertTrue(any(not samples for _, _, samples in calls[1:]))
            self.assertFalse(master.training)
            self.assertTrue(all(not policy.training for policy in members))
            evaluation = evaluate(path / 'checkpoint.pt', episodes=1, output_root=output)
            with (evaluation / 'metrics.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
