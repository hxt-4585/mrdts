"""Epoch means a fixed number of complete Master-plus-Member environment steps."""

import importlib.util
import unittest
from pathlib import Path
import tempfile
import csv
import json
import subprocess
import sys

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional RL dependency is not installed')
class TestRLEpochs(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('methods.rl_baseline.epoch_runner'),
                             'A bounded-memory full-step epoch runner is required')
        import torch
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_member_epoch_calls_master_every_step_without_resetting_at_flush(self):
        import torch
        from methods.rl_baseline.adapter import DelayEnvironment
        from methods.rl_baseline.epoch_runner import EpochLearner, run_epoch
        from methods.rl_baseline.ppo import PPOSettings
        env = DelayEnvironment()
        learner = EpochLearner(env, hidden=32, settings=PPOSettings(epochs=1))
        # Constant nonzero policy makes a skipped command/reset detectable.
        with torch.no_grad():
            learner.master.mean[-1].weight.zero_()
            learner.master.mean[-1].bias.fill_(.1)
        before = [p.detach().clone() for p in learner.master.parameters()]
        rows = []
        result = run_epoch(env, learner, stage='member', steps=3, update_every=2,
                           on_step=rows.append)
        self.assertEqual(result['steps'], 3)
        self.assertEqual(result['member_updates'], 2)
        self.assertEqual(result['master_updates'], 0)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row['slot_start'] for row in rows], [0., 1., 2.])
        self.assertEqual([row['master_calls'] for row in rows], [1, 1, 1])
        self.assertEqual([row['scheduled_tasks'] for row in rows], [1000] * 3)
        delta = np.tanh(.1) * 5. * 3
        np.testing.assert_allclose(env.members.positions[:12, :2],
                                   env.initial_positions[:12, :2] + delta, atol=2e-4)
        self.assertAlmostEqual(result['epoch_return'], sum(row['reward'] for row in rows))
        self.assertAlmostEqual(result['mean_reward'], result['epoch_return'] / 3)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, learner.master.parameters())))

    def test_master_epoch_uses_true_terminal_and_preserves_member(self):
        import torch
        from methods.rl_baseline.adapter import DelayEnvironment
        from methods.rl_baseline.epoch_runner import EpochLearner, run_epoch, master_critic_input
        from methods.rl_baseline.ppo import PPOSettings
        env = DelayEnvironment()
        learner = EpochLearner(env, hidden=32, settings=PPOSettings(epochs=1))
        before = [p.detach().clone() for p in learner.member.parameters()]
        state = master_critic_input(env, remaining=1.)
        np.testing.assert_array_equal(state[:, -1], np.ones(4))
        result = run_epoch(env, learner, stage='master', steps=3, update_every=2)
        self.assertEqual(result['member_updates'], 0)
        self.assertEqual(result['master_updates'], 1)
        self.assertEqual(result['steps'], 3)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, learner.member.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'epoch.pt'
            learner.save(path, {'epoch': 1})
            other = EpochLearner(env, hidden=32, settings=PPOSettings(epochs=1))
            self.assertEqual(other.load(path)['epoch'], 1)
            for a, b in zip(learner.master_value.parameters(), other.master_value.parameters()):
                torch.testing.assert_close(a, b)

    def test_cli_default_epoch_length_is_500_not_optimizer_passes(self):
        from methods.rl_baseline.epoch_training import parser
        args = parser().parse_args([])
        self.assertEqual(args.steps_per_epoch, 500)
        self.assertEqual(args.ppo_epochs, 4)

    def test_cli_logs_full_steps_resumes_and_evaluates(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            command = [sys.executable, str(root / 'scripts/train_rl_epochs.py'), '--output', str(output),
                       '--member-epochs', '1', '--master-epochs', '1', '--steps-per-epoch', '3',
                       '--update-every-steps', '2', '--eval-steps', '1', '--test-steps', '1',
                       '--hidden', '32', '--ppo-epochs', '1']
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            with (output / 'epochs.csv').open() as file:
                epochs = list(csv.DictReader(file))
            with (output / 'steps.csv').open() as file:
                steps = list(csv.DictReader(file))
            self.assertEqual([int(row['steps']) for row in epochs], [3, 3])
            self.assertEqual([int(row['global_step']) for row in steps], list(range(1, 7)))
            for epoch in epochs:
                selected = [row for row in steps if row['epoch'] == epoch['epoch']]
                self.assertAlmostEqual(float(epoch['epoch_return']), sum(float(row['reward']) for row in selected))
            # Simulate one logged step of an interrupted, uncheckpointed epoch.
            partial = dict(steps[-1], epoch='3', stage='master', stage_epoch='2', global_step='7', step='1')
            with (output / 'steps.csv').open('a', newline='') as file:
                csv.DictWriter(file, list(steps[0])).writerow(partial)
            command = [sys.executable, str(root / 'scripts/train_rl_epochs.py'), '--resume', str(output / 'latest.pt'),
                       '--member-epochs', '1', '--master-epochs', '2']
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['progress']['total_steps'], 9)
            with (output / 'steps.csv').open() as file:
                resumed = list(csv.DictReader(file))
            self.assertEqual([int(row['global_step']) for row in resumed], list(range(1, 10)))
            command = [sys.executable, str(root / 'scripts/evaluate_rl_epochs.py'),
                       '--checkpoint', str(output / 'best.pt')]
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn('mean_reward', completed.stdout)

    def test_reward_plot_reads_epoch_sum_and_labels_legacy_rounds(self):
        from methods.rl_baseline.epoch_reward_plot import read_rewards
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'config.json').write_text(json.dumps({'training': {'seed': 7, 'steps_per_epoch': 500}}))
            (folder / 'epochs.csv').write_text('epoch,stage,steps,epoch_return,mean_reward\n1,member,500,-250,-0.5\n')
            record = read_rewards(folder)
            self.assertFalse(record['legacy'])
            self.assertEqual(record['steps'], [500])
            self.assertEqual(record['returns'], [-250.])
            (folder / 'epochs.csv').unlink()
            (folder / 'config.json').write_text(json.dumps({'training': {'seed': 7, 'rollout_slots': 8}}))
            (folder / 'metrics.csv').write_text('stage,update,reward\ninitial_member,0,-1\nmember,1,-0.5\nmaster,1,-0.4\n')
            record = read_rewards(folder)
            self.assertTrue(record['legacy'])
            self.assertEqual(record['means'], [-.5, -.4])
            self.assertEqual(record['returns'], [-4., -3.2])
            self.assertEqual(record['epochs'], [1, 2])


if __name__ == '__main__':
    unittest.main()
