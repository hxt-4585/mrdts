"""Training artifacts, paired deployment and exact completed-epoch resume."""

import csv
from dataclasses import replace
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.config import load_config


class TestPPOExperiments(unittest.TestCase):
    def config(self, root, member=1, master=1):
        try:
            importlib.import_module('methods.solutions.ppo_delay.trainer')
        except ModuleNotFoundError:
            self.fail('Integrated PPO trainer is missing')
        return replace(load_config('config/experiments/ppo_delay.toml'), output_root=Path(root),
                       episodes=member+master, users=4, dag_nodes=3, slots=2, device='cpu',
                       training=dict(member_epochs=member, master_epochs=master, update_every_steps=1,
                                     eval_steps=2, test_steps=2, eval_every=1, hidden=16,
                                     ppo_epochs=1, threads=1, log_every_steps=100))

    def test_train_checkpoint_evaluation_and_random_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            from methods.solutions.ppo_delay.trainer import PPODelayTrainer
            from experiments.runner import evaluate
            path = PPODelayTrainer().train(config, torch.device('cpu'))
            self.assertEqual(path.parent.name, 'seed_42')
            self.assertEqual(path.parent.parent.name, 'ppo_delay_ers_ppo_master_ppo_member')
            self.assertEqual(json.loads((path/'metadata.json').read_text())['status'], 'completed')
            with (path/'training/epochs.csv').open() as stream:
                epochs = list(csv.DictReader(stream))
            with (path/'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(epochs), 2)
            self.assertEqual(len(rows), 4)
            self.assertEqual([int(r['episode_id']) for r in rows], [2,2,3,3])
            for epoch in epochs:
                selected = [r for r in rows if r['epoch']==epoch['epoch']]
                self.assertAlmostEqual(float(epoch['epoch_return']), sum(float(r['reward']) for r in selected))
            checkpoint = path/'checkpoints/best.pt'
            from experiments.cli import parse_config
            _, _, restored = parse_config(['--checkpoint', str(checkpoint)])
            self.assertEqual(restored.users, config.users)
            self.assertEqual(restored.episode_start, 1)
            self.assertEqual(restored.slots, config.training['test_steps'])
            _, _, resumed_config = parse_config(['--resume', str(path/'checkpoints/latest.pt'),
                                                '--master-epochs', '2'], training=True)
            self.assertEqual(resumed_config.training['hidden'], 16)
            self.assertEqual(resumed_config.episodes, 3)
            self.assertEqual(resumed_config.slots, 2)
            test_config = replace(config, episodes=1, slots=2, episode_start=1, checkpoint=checkpoint)
            with self.assertRaisesRegex(ValueError, 'held-out test split'):
                evaluate(replace(test_config, episodes=2))
            with self.assertRaisesRegex(ValueError, 'held-out test split'):
                evaluate(replace(test_config, episode_start=0))
            deployed = evaluate(test_config)
            report = json.loads((deployed/'summary.json').read_text())
            saved_test = json.loads((path/'training/test.json').read_text())
            self.assertAlmostEqual(report['truncated_mean_delay_s'], saved_test['mean_delay_s'])
            self.assertEqual(json.loads((deployed/'metadata.json').read_text())['mode'], 'evaluate')
            from experiments.aggregate import aggregate
            with aggregate(path.parents[4]).open() as stream:
                comparison = list(csv.DictReader(stream))
            self.assertEqual(len(comparison), 1, 'Training must not enter evaluation comparisons')

    def test_resume_matches_uninterrupted_and_archives_uncheckpointed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, member=2, master=1)
            from methods.solutions.ppo_delay.trainer import PPODelayTrainer
            from methods.solutions.ppo_delay.checkpoint import read_checkpoint
            trainer = PPODelayTrainer()
            full = trainer.train(config, torch.device('cpu'))
            short = trainer.train(replace(config, episodes=1, training={**config.training, 'member_epochs':1,'master_epochs':0}), torch.device('cpu'))
            with (short/'metrics.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            extra={**rows[-1], 'epoch':'2','global_step':'3'}
            with (short/'metrics.csv').open('a',newline='') as stream:
                csv.DictWriter(stream,list(extra)).writerow(extra)
            # A crash may replace a selected model before latest.pt commits that epoch.
            corrupt = read_checkpoint(short/'checkpoints/member_best.pt')
            for tensor in corrupt['member'].values():
                tensor.zero_()
            torch.save(corrupt,short/'checkpoints/member_best.pt')
            # Transition validation may flush before its checkpoint is committed.
            with (short/'training/validation.csv').open() as stream:
                validation = list(csv.DictReader(stream))
            with (short/'training/validation.csv').open('a', newline='') as stream:
                csv.DictWriter(stream,list(validation[-1])).writerow({**validation[-1], 'stage':'master'})
            resumed = trainer.train(replace(config,resume=short/'checkpoints/latest.pt'), torch.device('cpu'))
            self.assertEqual(short,resumed)
            a,b=read_checkpoint(full/'checkpoints/latest.pt'),read_checkpoint(resumed/'checkpoints/latest.pt')
            for name in ('member','master','member_value','master_value'):
                for key in a[name]:
                    torch.testing.assert_close(a[name][key],b[name][key],rtol=0,atol=0)
            self.assertEqual((full/'metrics.csv').read_text(),(resumed/'metrics.csv').read_text())
            self.assertTrue(list(resumed.glob('metrics.uncheckpointed-*.csv')))
            self.assertTrue(list((resumed/'training').glob('validation.uncheckpointed-*.csv')))
            from visualization.readers import read_run
            read_run(resumed)  # No duplicated stage-boundary validation rows after recovery.
            with self.assertRaisesRegex(ValueError,'frozen'):
                trainer.train(replace(config,episodes=4,training={**config.training,'member_epochs':3},resume=resumed/'checkpoints/latest.pt'),torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
