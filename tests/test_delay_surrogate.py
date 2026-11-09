import csv
import json
import tempfile
import tomllib
import unittest

import numpy as np
import torch

from methods.random_uav_ppo.support import settings_from_snapshot
from methods.hmappo.support import build_scene
from methods.local_surrogate.planner import LocalPlanner, capture_views
from methods.delay_surrogate.planner import DelayPlanner
from methods.delay_surrogate.run import ROOT, train
from methods.delay_surrogate.evaluate import compare


class TestDelaySurrogate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with (ROOT / 'config/methods/delay_surrogate.toml').open('rb') as stream:
            cls.config = tomllib.load(stream)['train']
        cls.source = torch.load(ROOT / cls.config['member_checkpoint'], map_location='cpu', weights_only=True)

    def scene(self):
        return build_scene(settings_from_snapshot(self.source['environment']), episode=0)

    def fitted_planner(self, env):
        planner = DelayPlanner(env, candidates=8, rounds=1, features=4, seed=45)
        views = capture_views(env, np.zeros(4))
        rng = np.random.default_rng(3)
        for _ in range(20):
            actions = rng.uniform(-1., 1., (12, 2))
            for view, model in zip(views, planner.models):
                model.update(view.features(actions), np.array([2. + .2 * actions[view.owned, 0].sum()]))
        return planner, views

    def test_scores_and_actions_ignore_all_energy_queues(self):
        env = self.scene()
        planner, views = self.fitted_planner(env)
        proposals = planner.candidates(views[0], np.zeros((12, 2)), 0)
        scores = planner.cost(0, views[0], proposals)
        planner.reset(9)
        first, _ = planner.actions(views)
        for view in views:
            view.queues[:] = np.arange(12) * 1e8
        np.testing.assert_array_equal(planner.cost(0, views[0], proposals), scores)
        planner.reset(9)
        second, _ = planner.actions(views)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(all(model.weights.shape[1] == 1 for model in planner.models))

    def test_delay_score_matches_energy_version_with_zero_queue(self):
        env = self.scene()
        planner, views = self.fitted_planner(env)
        previous = LocalPlanner(env, candidates=8, rounds=1, features=4, seed=45)
        rng = np.random.default_rng(3)
        for _ in range(20):
            actions = rng.uniform(-1., 1., (12, 2))
            for view, model in zip(views, previous.models):
                delay = 2. + .2 * actions[view.owned, 0].sum()
                model.update(view.features(actions), np.concatenate(([delay], np.ones(12))))
        proposals = planner.candidates(views[0], np.zeros((12, 2)), 0)
        np.testing.assert_allclose(planner.cost(0, views[0], proposals),
                                   previous.cost(0, views[0], proposals), atol=1e-12)

    def test_no_hidden_state_or_candidate_simulations(self):
        env = self.scene()
        env.masters.neighbor_region_ids = {1: (2,), 2: (1, 3), 3: (2, 4), 4: (3,)}
        planner, views = self.fitted_planner(env)
        proposals = planner.candidates(views[0], np.zeros((12, 2)), 0)
        before = planner.cost(0, views[0], proposals)
        hidden = ~views[0].visible
        env.members.positions[:12][hidden, :2] += 100.
        proposals[:, hidden] = .9
        changed = capture_views(env, np.zeros(4))[0]
        np.testing.assert_array_equal(before, planner.cost(0, changed, proposals))
        self.assertIsNone(env._runtime)
        self.assertEqual(env.step_index, 0)

    def test_short_pipeline_and_frozen_evaluation(self):
        config = dict(self.config, episodes=1, slots=2, candidates=4, rounds=1,
                      features=4, warmup_slots=1, log_interval=1)
        with tempfile.TemporaryDirectory(prefix='delay_surrogate_smoke_') as directory:
            config['output_root'] = directory
            output = train(**config)
            self.assertEqual(output.parent.parent.name, 'delay_surrogate')
            saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
            self.assertEqual(saved['method'], 'delay_surrogate')
            self.assertEqual(saved['slots_completed'], 2)
            for key, value in self.source['member_policies'].items():
                self.assertTrue(torch.equal(saved['member_policies'][key], value), key)
            for model in saved['surrogates']:
                self.assertEqual(model['weights'].shape[1], 1)
                self.assertEqual(model['samples'], 2)
            with (output / 'slots.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                self.assertNotIn('lyapunov_cost', row)
                self.assertAlmostEqual(float(row['delay_cost']), float(row['delay_sum_s'])
                                       / saved['environment']['env']['scheduling_duration_s'])
            evaluated = compare(output / 'checkpoint.pt', episodes=1, slots=1, output_root=directory)
            summary = json.loads((evaluated / 'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['surrogates_frozen'])
            self.assertEqual(set(summary['policies']), {'delay_surrogate', 'random', 'hover'})
            for stats in summary['policies'].values():
                self.assertNotIn('lyapunov_cost_mean', stats)
                self.assertTrue(np.isfinite(stats['mean_delay_s']))
                self.assertEqual(stats['slots_completed'], 1)
            self.assertEqual(json.loads((output / 'metadata.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
