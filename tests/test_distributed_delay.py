import ast
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from methods.distributed_delay.simulator import FrozenSimulation
from methods.distributed_delay.planner import Frame, LocalProxy, coordinate, frames_from_env


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/random_uav_ppo/seed_45/train_20261103T162321_743467Z_84a29361/checkpoint.pt'


class DistributedDelayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_independent_execution_matches_frozen_reference(self):
        from methods.centralized_cem.run import initialize_worker, evaluate_job
        source = torch.load(SOURCE, weights_only=True, map_location='cpu')
        simulation = FrozenSimulation(source)
        targets = simulation.env.members.positions[:simulation.count, :2] / simulation.side
        targets = targets + np.array([.01, -.01], dtype=np.float32)
        before = copy.deepcopy(source['member_policies'])
        actual = simulation.episode(200000, 2, targets=targets)
        with tempfile.TemporaryDirectory(prefix='distributed_delay_oracle_') as temporary:
            frozen = dict(source, random_flight=dict(flight_seed=source['config']['flight_seed'],
                                                   resample_slots=source['config']['resample_slots']))
            path = Path(temporary) / 'members.pt'
            torch.save(frozen, path)
            initialize_worker(path)
            expected = evaluate_job((targets, 200000, 2, 'learned'))
        self.assertAlmostEqual(actual['mean_delay_s'], expected['mean_delay_s'], places=12)
        self.assertEqual(actual['failed_dags'], expected['failed_dags'])
        for key in before:
            torch.testing.assert_close(source['member_policies'][key], before[key], rtol=0, atol=0)

    def test_hidden_positions_and_energy_do_not_enter_frames(self):
        simulation = FrozenSimulation(torch.load(SOURCE, weights_only=True, map_location='cpu'))
        env = simulation.env
        env.reset(epoch_index=0)
        frames = frames_from_env(env)
        for owner, frame in enumerate(frames):
            np.testing.assert_array_equal(frame.xy[~frame.visible], 0.)
            region = int(env.masters.region_ids[owner])
            expected = env.users.positions[env.users.region_ids == region, :2] / simulation.side
            np.testing.assert_allclose(frame.users, expected)
        env.members.energy_virtual_queue_j[:] = 1e12
        repeated = frames_from_env(env)
        for original, changed in zip(frames, repeated):
            np.testing.assert_array_equal(original.xy, changed.xy)
            np.testing.assert_array_equal(original.users, changed.users)

    def test_neighbor_gradient_is_combined_and_projection_is_feasible(self):
        class Quadratic:
            def __init__(self, target):
                self.target = torch.tensor(target, dtype=torch.float32)

            def value(self, frame, endpoint):
                return ((endpoint - self.target).square() * torch.tensor(frame.visible)[:, None]).sum()

        xy = np.array([[.02, .5], [.8, .5]], dtype=np.float32)
        frames = [Frame(xy.copy(), np.ones(2, bool), np.array([True, False]),
                        np.array([[.2, .5]], dtype=np.float32), (1,), .05),
                  Frame(xy.copy(), np.ones(2, bool), np.array([False, True]),
                        np.array([[.8, .5]], dtype=np.float32), (0,), .05)]
        models = [Quadratic([[.02, .5], [.8, .5]]), Quadratic([[.8, .5], [.8, .5]])]
        actions, stats = coordinate(frames, models, np.zeros_like(xy), rounds=3, rate=.02)
        self.assertGreater(actions[0, 0], 0.)
        self.assertTrue(np.all(np.abs(actions) <= 1.))
        endpoints = xy + actions * .05
        self.assertTrue(np.all((endpoints >= 0.) & (endpoints <= 1.)))
        self.assertEqual(stats['gradient_messages'], 6)
        self.assertEqual(stats['proxy_evaluations'], 6)

    def test_proxy_learns_position_cost_and_retains_action_gradient(self):
        frame = Frame(np.full((2, 2), .5, np.float32), np.ones(2, bool),
                      np.ones(2, bool), np.array([[.25, .75]], np.float32), (), .02)
        model = LocalProxy(2, frame.users, hidden=24, learning_rate=.003, replay_size=256)
        rng = np.random.default_rng(42)
        endpoints = rng.uniform(.1, .9, (160, 2, 2)).astype(np.float32)
        labels = .2 + ((endpoints - [.25, .75]) ** 2).sum((1, 2))
        for endpoint, target in zip(endpoints, labels):
            model.remember(frame, endpoint, float(target))
        initial = model.fit(steps=1, batch_size=64)
        final = model.fit(steps=160, batch_size=64)
        self.assertLess(final, initial / 3)
        endpoint = torch.tensor([[[.8, .2], [.8, .2]]], requires_grad=True)
        gradient = torch.autograd.grad(model.value(frame, endpoint).sum(), endpoint)[0]
        self.assertGreater(float(gradient[..., 0].mean()), 0.)
        self.assertLess(float(gradient[..., 1].mean()), 0.)

    def test_no_imports_from_other_baselines(self):
        for path in (ROOT / 'methods/distributed_delay').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('methods.'):
                    self.assertTrue(node.module.startswith('methods.distributed_delay'), (path, node.module))

    def test_training_resumes_exactly_without_reading_cem_during_learning(self):
        import json
        import tomllib
        from methods.distributed_delay.run import train
        with (ROOT / 'config/methods/distributed_delay.toml').open('rb') as stream:
            config = tomllib.load(stream)['train']
        config.update(episodes=1, train_slots=3, evaluation_slots=3, workers=1,
                      validation_interval=1, validation_epochs=[1500000], test_epochs=[2500000],
                      hidden=12, fit_steps=1, fit_interval=1, warmup_samples=1, replay_size=16,
                      batch_size=4, final_test=False, cem_reference='not-readable-during-training.pt')
        with tempfile.TemporaryDirectory(prefix='distributed_delay_smoke_') as temporary:
            directory = Path(temporary)
            first = train(config, output=directory / 'first')
            full = train(dict(config, episodes=2), output=directory / 'full')
            resumed = train(dict(config, episodes=2), output=directory / 'resumed', resume=first)
            expected = torch.load(full / 'latest.pt', weights_only=True)
            actual = torch.load(resumed / 'latest.pt', weights_only=True)
            for a, b in zip(actual['models'], expected['models']):
                for key in a['model']:
                    torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertFalse((resumed / 'test_results.json').exists())
            self.assertEqual(json.loads((resumed / 'metadata.json').read_text())['status'], 'completed')

    def test_paired_comparison_checks_uncertainty_not_only_mean(self):
        from methods.distributed_delay.evaluate import compare
        learned = [dict(mean_delay_s=x, failure_rate=0.) for x in (1.5, 100.5)]
        reference = [dict(mean_delay_s=x, failure_rate=0.) for x in (1., 100.)]
        result = compare(learned, reference, seed=45)
        self.assertTrue(result['point_within_1pct'])
        self.assertFalse(result['ci_supports_1pct'])

    def test_bootstrap_seed_is_independent_and_controlled_by_method_seed(self):
        from methods.distributed_delay.evaluate import compare
        learned = [dict(mean_delay_s=1. + index ** 2 / 100., failure_rate=0.)
                   for index in range(11)]
        reference = [dict(mean_delay_s=1., failure_rate=0.) for _ in learned]
        np.random.seed(123)
        expected_next = np.random.random(8)
        np.random.seed(123)
        first = compare(learned, reference, seed=45)
        repeated = compare(learned, reference, seed=45)
        changed = compare(learned, reference, seed=46)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first['bootstrap_seed'], changed['bootstrap_seed'])
        self.assertNotEqual(first['paired_delay_difference_95_ci_s'],
                            changed['paired_delay_difference_95_ci_s'])
        np.testing.assert_array_equal(np.random.random(8), expected_next)

    def test_monitor_writes_terminal_summary(self):
        import json
        import os
        from methods.distributed_delay.monitor import inspect
        with tempfile.TemporaryDirectory(prefix='distributed_delay_monitor_') as temporary:
            run = Path(temporary)
            (run / 'metadata.json').write_text(json.dumps(dict(status='completed', phase='done',
                                                               episode=2, pid=os.getpid())))
            (run / 'config.json').write_text(json.dumps(dict(training=dict(episodes=2))))
            result = inspect(run)
            self.assertTrue(result['process_alive'])
            self.assertEqual(result['status'], 'completed')
            self.assertTrue((run / 'completion.json').exists())

    def test_projected_moves_survive_float32_physical_integration(self):
        class Outward:
            def value(self, frame, endpoint):
                return (endpoint[..., 1] - endpoint[..., 0]).sum()
        physical = np.array([[999.99994, .00003], [999.99988, .00001]], dtype=np.float32)
        frame = Frame(physical / 1000., np.ones(2, bool), np.ones(2, bool),
                      np.array([[.5, .5]], np.float32), (), .005)
        actions, _ = coordinate([frame], [Outward()], np.zeros_like(physical), rounds=3, rate=.02)
        integrated = physical + (actions.astype(np.float32) * 10.) * .5
        self.assertTrue(np.all((integrated >= 0.) & (integrated <= 1000.)))

    def test_final_paired_evaluation_and_multiprocess_workers(self):
        import json
        import tomllib
        from methods.distributed_delay.run import train
        with (ROOT / 'config/methods/distributed_delay.toml').open('rb') as stream:
            config = tomllib.load(stream)['train']
        config.update(episodes=1, train_slots=2, evaluation_slots=2, workers=2,
                      validation_interval=1, validation_epochs=[1500000, 1500001],
                      test_epochs=[2500000, 2500001], hidden=12, fit_steps=1, fit_interval=1,
                      warmup_samples=1, replay_size=16, batch_size=4, final_test=True)
        with tempfile.TemporaryDirectory(prefix='distributed_delay_final_') as temporary:
            run = train(config, output=Path(temporary) / 'run')
            result = json.loads((run / 'test_results.json').read_text())
            learned = result['episodes']['distributed_delay']
            cem = result['episodes']['cem']
            self.assertEqual([row['epoch'] for row in learned], config['test_epochs'])
            self.assertEqual([row['dag_count'] for row in learned], [row['dag_count'] for row in cem])
            self.assertTrue(all(row['flight_violations'] == 0 for row in learned))
            self.assertIn('paired_1pct_margin_95_ci_s', result['comparison'])
            expected_seed = int(np.random.SeedSequence([config['seed'], 103]).generate_state(1)[0])
            self.assertEqual(result['comparison']['bootstrap_seed'], expected_seed)


if __name__ == '__main__':
    unittest.main()
