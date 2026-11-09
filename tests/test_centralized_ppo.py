import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from methods.centralized_ppo.policy import JointPPO
from methods.centralized_ppo.simulation import Simulator, flight_actions


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/random_uav_ppo/seed_45/train_20261103T162321_743467Z_84a29361/checkpoint.pt'


class TestJointPPO(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_joint_probability_and_mirrored_sampling(self):
        agent = JointPPO(np.zeros(6), np.full((3, 2), .5), hidden=16, learning_rate=.003)
        actions, old_mean = agent.sample(8, .4)
        torch.testing.assert_close(actions[:8] + actions[8:], 2 * old_mean.expand(8, -1))
        expected = torch.distributions.Normal(old_mean, .4).log_prob(actions).sum(-1)
        torch.testing.assert_close(agent.log_prob(actions, .4), expected)

    def test_policy_learns_joint_terminal_reward(self):
        agent = JointPPO(np.zeros(6), np.full((3, 2), .5), hidden=16, learning_rate=.003)
        target = torch.tensor([.65, .35] * 3)
        initial_error = float((torch.from_numpy(agent.targets()).flatten() - target).square().mean())
        for step in range(100):
            actions, old_mean = agent.sample(12, .4)
            rewards = -(actions.sigmoid() - target).square().sum(-1)
            stats = agent.update(actions, rewards, old_mean, .4, epochs=6, clip_ratio=.2, target_kl=.04)
            self.assertLessEqual(stats['kl'], .04001)
        error = float((torch.from_numpy(agent.targets()).flatten() - target).square().mean())
        self.assertLess(error, initial_error / 10)

    def test_deterministic_policy_survives_save(self):
        agent = JointPPO(np.arange(6) / 6, np.array([[.2, .8]]), hidden=16, learning_rate=.003)
        before = agent.targets()
        saved = copy.deepcopy(agent.state_dict())
        other = JointPPO(np.arange(6) / 6, np.array([[.2, .8]]), hidden=16, learning_rate=.003)
        other.load_state_dict(saved)
        np.testing.assert_array_equal(other.targets(), before)


class TestIndependentSimulation(unittest.TestCase):
    def test_actor_observation_contains_only_initial_xy_coordinates(self):
        source = torch.load(SOURCE, weights_only=True, map_location='cpu')
        simulator = Simulator(source)
        expected = np.concatenate((simulator.initial.ravel(),
            simulator.env.users.positions[:, :2].ravel() / simulator.side)).astype(np.float32)
        self.assertEqual(simulator.observation.shape, (84,))
        np.testing.assert_array_equal(simulator.observation, expected)

    def test_speed_limited_controller_stops_at_destination(self):
        positions = np.array([[5., 900.], [500., 500.]], dtype=np.float32)
        targets = np.array([[.8, .1], [.502, .499]], dtype=np.float32)
        for _ in range(180):
            actions = flight_actions(positions, targets, 1000., 5.)
            positions += actions * 5.
            self.assertTrue(np.all((positions >= 0.) & (positions <= 1000.)))
        np.testing.assert_allclose(positions, targets * 1000., atol=1e-4)
        np.testing.assert_array_equal(flight_actions(positions, targets, 1000., 5.), 0.)

    def test_frozen_scheduler_matches_reference_on_paired_stream(self):
        # Reference is used only as an independent test oracle, never in the new baseline.
        from methods.centralized_cem.run import initialize_worker, evaluate_job
        source = torch.load(SOURCE, weights_only=True, map_location='cpu')
        simulator = Simulator(source)
        weights = {k: v.clone() for k, v in source['member_policies'].items()}
        targets = simulator.initial.copy()
        targets += np.array([.005, -.005], dtype=np.float32)
        actual = simulator.evaluate(targets, 200000, 2)
        with tempfile.TemporaryDirectory(prefix='centralized_ppo_oracle_') as directory:
            frozen = dict(source, random_flight=dict(flight_seed=source['config']['flight_seed'],
                          resample_slots=source['config']['resample_slots']))
            path = Path(directory) / 'frozen.pt'
            torch.save(frozen, path)
            initialize_worker(path)
            expected = evaluate_job((targets, 200000, 2, 'learned'))
        self.assertAlmostEqual(actual['mean_delay_s'], expected['mean_delay_s'], places=12)
        self.assertEqual(actual['failed_dags'], expected['failed_dags'])
        self.assertEqual(actual['flight_violations'], 0)
        self.assertEqual(actual['slots_completed'], 2)
        for key, value in weights.items():
            torch.testing.assert_close(source['member_policies'][key], value, rtol=0, atol=0)
        repeated = simulator.evaluate(targets, 200000, 2)
        self.assertEqual(actual, repeated)

    def test_production_has_no_other_baseline_imports(self):
        for path in (ROOT / 'methods/centralized_ppo').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('methods.'):
                    self.assertTrue(node.module.startswith('methods.centralized_ppo'), (path, node.module))


class TestTrainingPipeline(unittest.TestCase):
    def test_monitor_records_completion_and_summary(self):
        from methods.centralized_ppo.monitor import inspect_run
        with tempfile.TemporaryDirectory(prefix='centralized_ppo_monitor_') as directory:
            run = Path(directory)
            (run / 'metadata.json').write_text(json.dumps(dict(status='completed', pid=os.getpid(), iteration=2)))
            (run / 'config.json').write_text(json.dumps(dict(updates=2)))
            (run / 'test_results.json').write_text(json.dumps(dict(comparisons={'cem': {'learned_mean_delay_s': .2}})))
            snapshot = inspect_run(run)
            self.assertEqual(snapshot['training_status'], 'completed')
            self.assertTrue(snapshot['process_alive'])
            self.assertTrue((run / 'monitor.json').exists())
            self.assertTrue((run / 'completion.txt').exists())

    def test_noninferiority_bootstrap_resamples_the_paired_reference_margin(self):
        from methods.centralized_ppo.run import paired_comparison
        learned = [dict(mean_delay_s=x, failed_dags=0) for x in (1.5, 100.5)]
        reference = [dict(mean_delay_s=x, failed_dags=0) for x in (1., 100.)]
        comparison = paired_comparison(learned, reference)
        self.assertTrue(comparison['point_estimate_within_1pct'])
        self.assertFalse(comparison['bootstrap_upper_within_1pct'])
        self.assertGreater(comparison['paired_1pct_margin_95_ci_s'][1], 0.)

    def test_resume_matches_uninterrupted_and_selects_validated_actor(self):
        from methods.centralized_ppo.run import train
        import tomllib
        with (ROOT / 'config/methods/centralized_ppo.toml').open('rb') as stream:
            config = tomllib.load(stream)['train']
        with tempfile.TemporaryDirectory(prefix='centralized_ppo_smoke_') as directory:
            directory = Path(directory)
            config.update(updates=2, pairs=2, workers=2, train_slots=2, evaluation_slots=2,
                          train_epoch_start=200000, train_epoch_count=2,
                          validation_epochs=[1000000], test_epochs=[2000000],
                          validation_interval=5, hidden=16, ppo_epochs=2, final_test=False)
            source = torch.load(SOURCE, weights_only=True, map_location='cpu')
            # Comparison is test-only: the optimizer must never consume these targets.
            simulator = Simulator(source)
            reference = directory / 'reference.pt'
            torch.save({'targets': torch.from_numpy(simulator.initial)}, reference)
            config['cem_reference'] = str(reference)
            first = train(config, output_dir=directory / 'first')
            full_config = dict(config, updates=3, final_test=True)
            full = train(full_config, output_dir=directory / 'full')
            resumed = train(full_config, output_dir=directory / 'resumed', resume_from=first)
            expected = torch.load(full / 'latest.pt', weights_only=True, map_location='cpu')
            actual = torch.load(resumed / 'latest.pt', weights_only=True, map_location='cpu')
            for key in expected['policy']:
                torch.testing.assert_close(actual['policy'][key], expected['policy'][key], rtol=0, atol=0)
            selected = torch.load(resumed / 'best.pt', weights_only=True, map_location='cpu')
            expected_selected = torch.load(full / 'best.pt', weights_only=True, map_location='cpu')
            self.assertEqual(selected['iteration'], expected_selected['iteration'])
            torch.testing.assert_close(selected['targets'], expected_selected['targets'], rtol=0, atol=0)
            self.assertFalse((first / 'test_results.json').exists())
            agent = JointPPO(simulator.observation, simulator.initial, hidden=16, learning_rate=config['learning_rate'])
            agent.load_state_dict(selected['policy'])
            np.testing.assert_array_equal(agent.targets(), selected['targets'].numpy())
            result = json.loads((resumed / 'test_results.json').read_text())
            self.assertEqual(result['episodes']['learned'][0]['epoch'], 2000000)
            self.assertEqual(result['episodes']['learned'][0]['slots_completed'], 2)
            self.assertEqual(json.loads((resumed / 'metadata.json').read_text())['status'], 'completed')
            frozen = torch.load(resumed / 'frozen_members.pt', weights_only=True, map_location='cpu')
            for key in source['member_policies']:
                torch.testing.assert_close(frozen['member_policies'][key], source['member_policies'][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
