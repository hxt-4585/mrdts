import json
from pathlib import Path
import tempfile
import time
import tomllib
import unittest

import numpy as np
import torch

from methods.centralized_cem.run import ROOT, cem_update, destination_actions, train


class TestCentralizedCEM(unittest.TestCase):
    def test_flight_reaches_destination_without_overshoot(self):
        positions = np.array([[1., 999.], [500., 500.]])
        targets = np.array([[.999, .001], [.502, .499]])
        for _ in range(200):
            actions = destination_actions(positions, targets, 1000., 5.)
            self.assertTrue(np.all(np.abs(actions) <= 1.))
            positions += actions * 5.
            self.assertTrue(np.all((positions >= 0.) & (positions <= 1000.)))
        np.testing.assert_allclose(positions, targets * 1000., atol=1e-5)

    def test_search_improves_known_objective(self):
        rng = np.random.default_rng(45)
        mean, std = np.zeros((12, 2)), np.ones((12, 2))
        for _ in range(50):
            candidates = rng.normal(mean, std, (64, 12, 2))
            costs = ((candidates - .3) ** 2).sum(axis=(1, 2))
            mean, std = cem_update(mean, std, candidates, costs, 8, .7, .005)
        self.assertLess(float(((mean - .3) ** 2).mean()), .001)

    def test_pipeline_and_frozen_members(self):
        with (ROOT / 'config/methods/centralized_cem.toml').open('rb') as stream:
            config = tomllib.load(stream)['train']
        with tempfile.TemporaryDirectory(prefix='centralized_cem_smoke_') as directory:
            source = torch.load(ROOT / config['member_checkpoint'], weights_only=True)
            original = {k: v.clone() for k, v in source['member_policies'].items()}
            # Only the test copy gets a short horizon; production keeps 500 slots.
            source['environment']['env']['steps_per_epoch'] = 2
            checkpoint = Path(directory) / 'source.pt'
            torch.save(source, checkpoint)
            config.update(member_checkpoint=str(checkpoint), output_root=directory,
                          generations=2, population=4, elites=2, workers=2,
                          train_slots=2, train_epochs=[200000], validation_epochs=[1000000],
                          test_epochs=[2000000, 2000001], validation_interval=1)
            started = time.perf_counter()
            output = train(config)
            saved = torch.load(output / 'checkpoint.pt', weights_only=True)
            for name, value in saved['member_policies'].items():
                self.assertTrue(torch.equal(value, original[name]), name)
            result = json.loads((output / 'test_results.json').read_text())
            self.assertFalse(result['training_plateau'])
            self.assertEqual(result['conclusion'], 'needs_followup')
            for rows in result['episodes'].values():
                self.assertTrue(all(row['slots_completed'] == 2 for row in rows))
                self.assertEqual([r['epoch'] for r in rows], [2000000, 2000001])
            self.assertTrue(all(row['flight_violations'] == 0 for row in result['episodes']['learned']))
            self.assertEqual(json.loads((output / 'metadata.json').read_text())['status'], 'completed')
            config['generations'] = 3
            uninterrupted = train(config)
            checkpoint.unlink()
            resumed = train(config, resume_from=output)
            expected = json.loads((uninterrupted / 'search_state.json').read_text())
            actual = json.loads((resumed / 'search_state.json').read_text())
            self.assertEqual(actual, expected)
            resumed_saved = torch.load(resumed / 'checkpoint.pt', weights_only=True)
            self.assertEqual(resumed_saved['generation'], 3)
            for name, value in resumed_saved['member_policies'].items():
                self.assertTrue(torch.equal(value, original[name]), name)
            self.assertEqual(torch.load(output / 'checkpoint.pt', weights_only=True)['generation'], 2)
            resumed_metadata = json.loads((resumed / 'metadata.json').read_text())
            self.assertEqual(resumed_metadata['resumed_from_generation'], 2)
            print(f'Smoke elapsed seconds: {time.perf_counter()-started:.2f}')


if __name__ == '__main__':
    unittest.main()
