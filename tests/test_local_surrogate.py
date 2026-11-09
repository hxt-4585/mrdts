import json
from pathlib import Path
import tempfile
import tomllib
import unittest

import numpy as np
import torch

from methods.hmappo.support import build_scene
from methods.local_surrogate.surrogate import OnlineSurrogate
from methods.local_surrogate.planner import LocalPlanner, capture_views
from methods.local_surrogate.run import ROOT, train
from methods.local_surrogate.evaluate import compare
from methods.random_uav_ppo.support import settings_from_snapshot
from methods.frozen_member_ppo.run import frozen_policies
from methods.hmappo.rollout import Adapter, collect_slot
from methods.local_surrogate.rollout import execute_slot, physical_target, regional_delays, lyapunov_cost


class TestOnlineSurrogate(unittest.TestCase):
    def test_learns_physical_outputs_and_roundtrips(self):
        rng = np.random.default_rng(3)
        model = OnlineSurrogate(2, 2, features=8, ridge=.01, seed=7)
        for _ in range(200):
            x = rng.uniform(-1., 1., 2)
            model.update(x, np.array([2. + x[0], 1. + .4 * x[1]]))
        x = rng.uniform(-1., 1., (20, 2))
        mean, uncertainty = model.predict(x)
        np.testing.assert_allclose(mean, np.column_stack((2. + x[:, 0], 1. + .4 * x[:, 1])), atol=.02)
        self.assertTrue(np.all(uncertainty >= 0.))
        restored = OnlineSurrogate.from_state(model.state())
        np.testing.assert_array_equal(restored.predict(x)[0], mean)


class TestLocalPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with (ROOT / 'config/methods/local_surrogate.toml').open('rb') as stream:
            cls.config = tomllib.load(stream)['train']
        cls.source = torch.load(ROOT / cls.config['member_checkpoint'], weights_only=True, map_location='cpu')

    def scene(self):
        return build_scene(settings_from_snapshot(self.source['environment']), episode=0)

    def test_view_ignores_hidden_positions_queues_and_actions(self):
        env = self.scene()
        # Use a chain to ensure the first Master's two-hop region is hidden.
        env.masters.neighbor_region_ids = {1: (2,), 2: (1, 3), 3: (2, 4), 4: (3,)}
        view = capture_views(env, np.zeros(4))[0]
        actions = np.zeros((12, 2))
        features = view.features(actions)
        hidden = ~view.visible
        self.assertTrue(hidden.any())
        env.members.positions[:12][hidden, :2] += 100.
        env.members.energy_virtual_queue_j[:12][hidden] += 10000.
        actions[hidden] = .9
        changed = capture_views(env, np.zeros(4))[0]
        np.testing.assert_array_equal(changed.features(actions), features)
        np.testing.assert_array_equal(changed.queues, view.queues)

    def test_search_preserves_ownership_and_does_not_step_environment(self):
        env = self.scene()
        planner = LocalPlanner(env, candidates=8, rounds=2, features=8, seed=3)
        views = capture_views(env, np.zeros(4))
        positions = env.members.positions.copy()
        rng_state = json.dumps(env.dag_generator._np_rng.bit_generator.state)
        actions, stats = planner.actions(views, explore=0.)
        np.testing.assert_array_equal(env.members.positions, positions)
        self.assertEqual(json.dumps(env.dag_generator._np_rng.bit_generator.state), rng_state)
        self.assertEqual(env.step_index, 0)
        self.assertIsNone(env._runtime)
        endpoints = positions[:12, :2] + planner.full_step * actions
        self.assertTrue(np.all(np.abs(actions) <= 1.))
        self.assertTrue(np.all((endpoints >= 0.) & (endpoints <= env.region.config.side_length)))
        for owner, view in enumerate(views):
            proposals = planner.candidates(view, actions, owner)
            np.testing.assert_array_equal(proposals[:, ~view.owned], np.broadcast_to(actions[~view.owned], proposals[:, ~view.owned].shape))
        self.assertGreater(stats['candidate_evaluations'], 0)

    def test_queue_weights_change_cost_without_retraining(self):
        env = self.scene()
        planner = LocalPlanner(env, candidates=8, features=8, seed=3, uncertainty_penalty=0.)
        view = capture_views(env, np.zeros(4))[0]
        model = planner.models[0]
        # Constant positive residual energy makes queue reweighting directly observable.
        model.weights.fill(0.)
        model.weights[0] = 1.
        actions = np.zeros((1, 12, 2))
        before = planner.cost(0, view, actions)
        view.queues[view.owned] = 10.
        after = planner.cost(0, view, actions)
        self.assertTrue(np.all(after > before))
        self.assertEqual(model.samples, 0)

    def test_frozen_rollout_matches_existing_low_layer(self):
        env = self.scene()
        _, members = frozen_policies(env, self.source['config']['hidden'],
                                     self.source['member_policies'], 'cpu')
        adapter = Adapter(env)
        actions = np.random.default_rng(8).uniform(-1., 1., (12, 2))
        _, _, existing = collect_slot(adapter, None, members, flight_actions=actions,
                                      member_deterministic=True)
        queues = env.members.energy_virtual_queue_j.copy()
        positions = env.members.positions.copy()
        env.reset(epoch_index=0)
        views = capture_views(env, np.zeros(4))
        result, row = execute_slot(adapter, members, actions)
        for key in ('mean_delay_s', 'total_energy_j', 'member_decisions', 'failure_rate'):
            self.assertAlmostEqual(row[key], existing[key], places=6)
        np.testing.assert_array_equal(env.members.positions, positions)
        np.testing.assert_allclose(env.members.energy_virtual_queue_j, queues, atol=1e-6)
        delays = regional_delays(result, env.users.region_ids, 4, env.env_config.scheduling_duration_s)
        energy = np.zeros(12)
        for view, delay in zip(views, delays):
            target = physical_target(view, result, delay, env.members.config.member_slot_energy)
            self.assertTrue(np.all(target[1:][~view.owned] == 0.))
            energy += target[1:]
        np.testing.assert_allclose(energy, (result.total_energy_j - result.flight_energy_j)
                                   / env.members.config.member_slot_energy)
        self.assertAlmostEqual(lyapunov_cost(views, result, delays, 145., 1.), delays.sum())

    def test_empty_region_can_receive_neighbor_proposals(self):
        env = self.scene()
        # A legitimate empty source region still learns local GD delay and returns quotes.
        owned = env.members.region_ids[:12] == 1
        recipient = int(env.masters.neighbor_region_ids[1][0])
        destination = env.masters.positions[recipient - 1, :2]
        env.members.positions[:12][owned, :2] = destination
        env._refresh_topology()
        planner = LocalPlanner(env, candidates=4, features=4, rounds=1)
        views = capture_views(env, np.zeros(4))
        self.assertFalse(views[0].owned.any())
        actions, _ = planner.actions(views)
        self.assertTrue(np.all(np.isfinite(actions)))

    def test_communication_counts_include_selected_plan_broadcasts(self):
        env = self.scene()
        planner = LocalPlanner(env, candidates=4, features=4, rounds=2)
        views = capture_views(env, np.zeros(4))
        _, stats = planner.actions(views)
        links = sum(map(len, planner.neighbors))
        self.assertEqual(stats.get('plan_messages', 0), 2 * links)
        self.assertEqual(stats['neighbor_messages'], 6 * links)
        _, exploring = planner.actions(views, explore=1.)
        self.assertEqual(exploring['plan_messages'], links)
        self.assertEqual(exploring['quote_messages'], 0)

    def test_pipeline_freezes_low_layer_and_evaluation_models(self):
        config = dict(self.config, episodes=1, slots=2, warmup_slots=1,
                      candidates=4, rounds=1, features=4, log_interval=1)
        with tempfile.TemporaryDirectory(prefix='local_surrogate_smoke_') as directory:
            config['output_root'] = directory
            output = train(**config)
            checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True, map_location='cpu')
            for key, value in self.source['member_policies'].items():
                self.assertTrue(torch.equal(checkpoint['member_policies'][key], value), key)
            self.assertEqual(checkpoint['slots_completed'], 2)
            self.assertTrue(all(state['samples'] == 2 for state in checkpoint['surrogates']))
            evaluated = compare(output / 'checkpoint.pt', episodes=1, slots=1,
                                output_root=directory, epoch_start=1000000)
            summary = json.loads((evaluated / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(set(summary['policies']), {'local_surrogate', 'random', 'hover'})
            self.assertTrue(summary['surrogates_frozen'])
            for row in summary['policies'].values():
                self.assertEqual(row['slots_completed'], 1)
                self.assertTrue(np.isfinite(row['mean_delay_s']))
                self.assertTrue(np.isfinite(row['lyapunov_cost_mean']))
            metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
