import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import torch

from methods.hmappo.ppo import Policy, Sample, update
from methods.hmappo.run import train
from methods.hmappo.diagnostics import explained_variance
from methods.hmappo.support import MetricSummary


class TestHmappoLogging(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def make_samples(self, policy):
        obs = torch.randn(6, 3)
        masks = torch.ones(6, 2, dtype=torch.bool)
        if policy.continuous:
            masks[-1] = False
        actions, logp, values = policy.sample(obs, obs, masks)
        return [Sample(o.numpy(), o.numpy(), m.numpy(), a.numpy(), lp.item(), v.item(),
                       reward=-i, advantage=i - 2., target=i * .3)
                for i, (o, m, a, lp, v) in enumerate(zip(obs, masks, actions, logp, values))]

    def test_diagnostics_do_not_change_parameters_or_random_stream(self):
        for continuous in (True, False):
            with self.subTest(continuous=continuous):
                policy = Policy(3, 3, 2, continuous, 8)
                samples = self.make_samples(policy)
                other = copy.deepcopy(policy)
                options = dict(ppo_epochs=2, minibatch_size=4, clip_ratio=.2,
                               entropy_coef=.01, value_coef=.5, max_grad_norm=.5)
                rng = torch.get_rng_state()
                stats = update(policy, torch.optim.Adam(policy.parameters()), samples,
                               diagnostics=True, **options)
                after = torch.get_rng_state()
                torch.set_rng_state(rng)
                update(other, torch.optim.Adam(other.parameters()), samples,
                       diagnostics=False, **options)
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                for key, value in policy.state_dict().items():
                    self.assertTrue(torch.equal(value, other.state_dict()[key]), key)
                self.assertEqual(stats['active_samples'], 5 if continuous else 6)
                self.assertAlmostEqual(stats['reward_mean'], -2.5)
                self.assertIsNotNone(stats['final_kl'])
                obs = torch.tensor(np.array([s.observation for s in samples]))
                masks = torch.tensor(np.array([s.mask for s in samples]))
                actions = torch.tensor(np.array([s.action for s in samples]))
                with torch.no_grad():
                    distribution = policy.distribution(obs, masks)
                    logp = distribution.log_prob(actions)
                    if continuous:
                        jacobian = 2 * (np.log(2.) - actions - torch.nn.functional.softplus(-2 * actions))
                        logp = ((logp - jacobian) * masks).sum(-1)
                    logratio = logp - torch.tensor([s.log_prob for s in samples])
                    active = masks.any(-1)
                    expected = (logratio.exp() - 1 - logratio)[active].mean().item()
                self.assertAlmostEqual(stats['final_kl'], expected, places=6)

    def test_csv_schema_handles_empty_members_and_skipped_diagnostics(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()) as console:
            path = train(episodes=1, slots=3, users=4, dag_nodes=3, hidden=8,
                         rollout_slots=1, ppo_epochs=1, output_root=root,
                         diagnostics_interval=2, log_interval=2)
            with (path / 'updates.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(None not in row for row in rows))
            for row in rows:
                self.assertNotEqual(row['master_final_kl'], '')
                for owner in range(12):
                    prefix = f'member_{owner}_'
                    if int(row[prefix + 'samples']) == 0:
                        self.assertEqual(row[prefix + 'reward_mean'], '')
                    if row['update'] == '1':
                        self.assertEqual(row[prefix + 'final_kl'], '')
            with (path / 'metrics.csv').open() as stream:
                slots = list(csv.DictReader(stream))
            summary = json.loads((path / 'summary.json').read_text())
            count = sum(int(row['dag_count']) for row in slots)
            self.assertEqual(summary['dag_count'], count)
            self.assertAlmostEqual(summary['mean_delay_s'],
                                   sum(float(row['delay_sum_s']) for row in slots) / count)
            with (path / 'episodes.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            self.assertEqual(sum(line.startswith('Update ') for line in console.getvalue().splitlines()), 2)

    def test_value_fit_and_weighted_streaming_summary(self):
        targets = torch.tensor([1., 2., 3.])
        self.assertEqual(explained_variance(targets, targets), 1.)
        self.assertEqual(explained_variance(torch.zeros(3), targets), 0.)
        self.assertIsNone(explained_variance(targets, torch.ones(3)))
        summary = MetricSummary()
        common = dict(flight_violations=1, master_reward=-1., master_training_reward_mean=-2.,
                      master_boundary_penalty_sum=4., member_reward_sum=-3., member_decisions=2)
        summary.add(dict(dag_count=1, failed_dags=0, delay_sum_s=2., successful_delay_sum_s=2., **common))
        summary.add(dict(dag_count=3, failed_dags=2, delay_sum_s=10., successful_delay_sum_s=4., **common))
        result = summary.result()
        self.assertEqual(result['mean_delay_s'], 3.)
        self.assertEqual(result['successful_mean_delay_s'], 3.)
        self.assertEqual(result['failure_rate'], .5)
        self.assertEqual(result['master_training_reward_mean'], -2.)
        self.assertEqual(result['member_reward_per_decision'], -1.5)
