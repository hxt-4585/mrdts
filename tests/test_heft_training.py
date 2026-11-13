"""Exercise learned flight, checkpoint restoration and actual HEFT rollouts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from methods.heft.run import load_config, train, evaluate
from methods.heft.rollout import make_settings, make_environment, run_slot
from methods.heft.flight import FlightPolicy


class TestHEFTTraining(unittest.TestCase):
    def config(self):
        config = load_config()
        config['train'].update(episodes=2, slots=2, samplers=1, rollout_slots=1,
                               users=4, dag_nodes=3, validation_interval=1,
                               evaluation_slots=1, validation_epochs=[1500000],
                               test_epochs=[2500000], checkpoint_interval=1)
        config['high'].update(hidden=8, replay_size=8, batch_size=2, fit_interval=1,
                              fit_steps=1, warmup_samples=1, rounds=1)
        return config

    def test_learning_and_frozen_evaluation_are_reproducible(self):
        torch.set_num_threads(1)
        config = self.config()
        settings = make_settings(config['train'])
        env = make_environment(settings)
        policy = FlightPolicy(env, config['high'])
        env.reset(epoch_index=400000)
        policy.reset(env.epoch_index)
        before = [p.detach().clone() for p in policy.networks.parameters()]
        row, experience = run_slot(env, policy, exploration=1.)
        loss = policy.learn([experience])
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, policy.networks.parameters())))
        self.assertEqual(row['member_decisions'], 12)
        state = policy.state()
        clone = FlightPolicy(env, config['high'])
        clone.restore(state)
        env.reset(epoch_index=400001)
        policy.reset(env.epoch_index)
        clone.reset(env.epoch_index)
        scene = policy.observe(env)
        self.assertTrue(np.allclose(policy.act(scene, 0.), clone.act(scene, 0.)))
        self.assertEqual(policy.samples, 1)

    def test_checkpoint_resume_matches_uninterrupted_training_and_evaluation(self):
        config = self.config()
        with tempfile.TemporaryDirectory(prefix='mrdts_heft_test_') as directory:
            root = Path(directory)
            full = train(copy.deepcopy(config), root / 'full')
            partial_config = copy.deepcopy(config)
            partial_config['train']['episodes'] = 2
            partial = train(partial_config, root / 'resumed', stop_after=1)
            resumed = train(copy.deepcopy(config), root / 'resumed', resume=partial)
            full_state = torch.load(full, weights_only=True)
            resumed_state = torch.load(resumed, weights_only=True)
            self.assertEqual(resumed_state['completed_episodes'], 2)
            for name, tensor in full_state['flight']['weights'].items():
                self.assertTrue(torch.equal(tensor, resumed_state['flight']['weights'][name]), name)
            report = evaluate(resumed, slots=1)
            repeated = evaluate(resumed, slots=1)
            # Wall-clock planning cost varies; simulation outcomes must not.
            for result in (report, repeated):
                result['summary'].pop('planning_seconds')
                for row in result['rows']:
                    row.pop('planning_seconds')
            self.assertEqual(report, repeated)
            self.assertEqual(report['summary']['dag_count'], 4)
            self.assertTrue(np.isfinite(report['summary']['mean_delay_s']))
            self.assertEqual(len((root / 'resumed' / 'episodes.jsonl').read_text().splitlines()), 2)
            json.dumps(report, allow_nan=False)

    def test_multiple_samplers_and_explicit_resume_overrides(self):
        config = self.config()
        config['train'].update(samplers=2, slots=3)
        with tempfile.TemporaryDirectory(prefix='mrdts_heft_test_') as directory:
            root = Path(directory)
            checkpoint = train(config, root, stop_after=1)
            checkpoint = train(config, root, resume=checkpoint, episodes=2, slots=2)
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved['completed_episodes'], 2)
            self.assertEqual(saved['flight']['samples'], 5)
            self.assertEqual(saved['config']['train']['slots'], 2)
            self.assertEqual(saved['environment']['env']['steps_per_epoch'], 2)
            lines = (root / 'slots.csv').read_text().splitlines()
            self.assertEqual(len(lines), 6)


if __name__ == '__main__':
    unittest.main()
