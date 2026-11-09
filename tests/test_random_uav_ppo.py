import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from methods.random_uav_ppo.support import build_scene, resolved_settings


class TestRandomUAVPPO(unittest.TestCase):
    def test_curve_aggregation_weights_samples_and_excludes_missing_and_partial(self):
        from methods.random_uav_ppo.plot_member import aggregate

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'episodes.csv').write_text('episode\n0\n')
            (root / 'updates.csv').write_text(
                'episode,member_0_samples,member_0_reward_mean,member_0_categorical_entropy_normalized\n'
                '0,2,-1,0.5\n0,6,-3,\n0,0,,\n1,100,-50,0\n')
            rows = aggregate(root, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['samples'], 8)
            self.assertEqual(rows[0]['reward'], -2.5)
            self.assertEqual(rows[0]['entropy'], 0.5)
            self.assertEqual(rows[0]['entropy_weight'], 2)
            self.assertIsNone(rows[0]['return'])

    def test_vector_persists_until_resampling(self):
        from methods.random_uav_ppo.flight import RandomFlight
        env = build_scene(resolved_settings(), episode=0)
        flight = RandomFlight(env, resample_slots=10)
        first = flight.actions(env, 0).copy()
        for slot in range(1, 10):
            np.testing.assert_array_equal(first, flight.actions(env, slot))
        self.assertFalse(np.array_equal(first, flight.actions(env, 10)))

    def test_random_vectors_bounds_and_reproducibility(self):
        from methods.random_uav_ppo.flight import RandomFlight
        env = build_scene(resolved_settings(), episode=0)
        flight = RandomFlight(env, flight_seed=46)
        traces = []
        for episode in (0, 1, 0):
            env.reset(epoch_index=episode)
            trace = []
            above_unit_norm = False
            for slot in range(500):
                action = flight.actions(env, slot)
                self.assertTrue(np.all(np.abs(action) <= 1.))
                above_unit_norm |= bool(np.any(np.linalg.norm(action, axis=1) > 1.))
                self.assertFalse(env.members.apply_flight_actions(action, env.region.config.side_length).any())
                trace.append(env.members.positions[:flight.count, :2].copy())
            self.assertTrue(above_unit_norm)
            traces.append(trace)
        np.testing.assert_array_equal(traces[0], traces[2])
        self.assertFalse(np.array_equal(traces[0], traces[1]))
        self.assertTrue(np.all(np.any(np.diff(traces[0], axis=0) != 0, axis=(0, 2))))
        # 在边缘与角点反复采样，直接交给真实环境检查边界行为。
        edge = env.region.config.side_length
        corners = np.array([[0., 0.], [edge, 0.], [0., edge], [edge, edge]])
        for slot in range(500):
            env.members.positions[:flight.count, :2] = np.resize(corners, (flight.count, 2))
            action = flight.actions(env, slot)
            self.assertFalse(env.members.apply_flight_actions(action, edge).any())

    def test_empty_regions_use_environment_local_execution(self):
        from methods.random_uav_ppo.rollout import Adapter, collect_slot
        env = build_scene(resolved_settings(users=4, dag_nodes=3), episode=0)
        env.members.positions[:env.members.member_uav_count, :2] = env.members.positions[0, :2]
        env._refresh_topology()
        adapter = Adapter(env)
        requests = []
        samples, metrics = collect_slot(adapter, None,
            member_selector=lambda key, mask: key.owner_member_id,
            request_observer=requests.extend)
        unowned = [request for request in requests if request.owner_member is None]
        self.assertTrue(unowned)
        self.assertGreater(metrics['flight_empty_regions'], 0)
        self.assertEqual(metrics['dag_count'], 4)
        self.assertEqual(sum(map(len, samples.values())), (4 - len(unowned)) * 3)
        self.assertEqual(metrics['flight_violations'], 0)

    def test_method_seed_controls_network_and_flight_without_overriding_environment(self):
        from methods.random_uav_ppo.run import train, load_checkpoint
        from methods.random_uav_ppo.flight import RandomFlight

        settings = resolved_settings(users=4, dag_nodes=3)
        settings['env'] = replace(settings['env'], env_seed=123)
        with tempfile.TemporaryDirectory() as root:
            runs = []
            for seed in (45, 45, 46):
                with patch('methods.random_uav_ppo.run.resolved_settings', return_value=dict(settings)):
                    out = train(episodes=1, slots=2, hidden=16, seed=seed,
                                ppo_epochs=1, rollout_slots=2, output_root=root)
                env, members, saved = load_checkpoint(out / 'initial.pt')
                flight = RandomFlight(env, flight_seed=saved['config']['flight_seed'])
                trained = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=True)
                runs.append((saved, members.state_dict(), flight.actions(env, 0).copy(),
                             env.users.positions.copy(), trained['member_policies']))
            for saved, _, _, _, _ in runs:
                self.assertEqual(saved['environment']['env']['env_seed'], 123)
            for key in runs[0][1]:
                self.assertTrue(torch.equal(runs[0][1][key], runs[1][1][key]))
                self.assertTrue(torch.equal(runs[0][4][key], runs[1][4][key]))
            self.assertTrue(any(not torch.equal(runs[0][1][key], runs[2][1][key])
                                for key in runs[0][1]))
            np.testing.assert_array_equal(runs[0][2], runs[1][2])
            self.assertFalse(np.array_equal(runs[0][2], runs[2][2]))
            np.testing.assert_array_equal(runs[0][3], runs[2][3])

    def test_flight_random_stream_is_isolated_from_policy_sampling(self):
        from methods.random_uav_ppo.flight import RandomFlight, derive_flight_seed

        env = build_scene(resolved_settings(), episode=0)
        first = RandomFlight(env, flight_seed=derive_flight_seed(45), resample_slots=1)
        second = RandomFlight(env, flight_seed=derive_flight_seed(45), resample_slots=1)
        for slot in range(3):
            expected = first.actions(env, slot).copy()
            torch.rand(1000)
            np.random.random(1000)
            np.testing.assert_array_equal(expected, second.actions(env, slot))

    def test_training_and_paired_evaluation(self):
        from methods.random_uav_ppo.run import train, load_checkpoint
        from methods.random_uav_ppo.evaluate import compare
        with tempfile.TemporaryDirectory() as root:
            out = train(episodes=2, slots=4, users=4, dag_nodes=3, hidden=16, seed=73,
                        ppo_epochs=1, rollout_slots=4, output_root=root,
                        metrics_interval=3, snapshot_interval=1)
            self.assertEqual(out.parent.parent.name, 'random_uav_ppo')
            self.assertEqual({p.name for p in out.glob('*.pt')},
                             {'initial.pt', 'checkpoint.pt', 'snapshot.pt'})
            _, _, saved = load_checkpoint(out / 'snapshot.pt')
            self.assertEqual(saved['episodes_completed'], 2)
            with (out / 'metrics.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            self.assertEqual(json.loads((out / 'metadata.json').read_text())['slots_completed'], 8)
            _, initial, _ = load_checkpoint(out / 'initial.pt')
            _, trained, checkpoint = load_checkpoint(out / 'checkpoint.pt')
            self.assertEqual(checkpoint['episodes_completed'], 2)
            self.assertTrue(any(not torch.equal(v, trained.state_dict()[k])
                                for k, v in initial.state_dict().items()))
            result = compare([out / 'initial.pt', out / 'checkpoint.pt'], episodes=2, output_root=root)
            with (result / 'metrics.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            with (result / 'episodes.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for episode in (1000000, 1000001):
                paired = [r for r in rows if int(r['episode']) == episode]
                self.assertEqual(len({r['dag_stream_sha256'] for r in paired}), 1)
                self.assertEqual(len({r['flight_stream_sha256'] for r in paired}), 1)
            self.assertEqual(len({r['flight_stream_sha256'] for r in rows}), 2)
            self.assertEqual(json.loads((out / 'trajectory.json').read_text())['policy'], 'random_vector')
            evaluated = json.loads((result / 'config.json').read_text())
            self.assertEqual(evaluated['seed'], 73)
            self.assertEqual(evaluated['flight_seed'], checkpoint['config']['flight_seed'])
            changed = compare([out / 'checkpoint.pt'], episodes=1, seed=46, output_root=root)
            self.assertEqual(changed.parent.name, 'seed_46')
            with (changed / 'episodes.csv').open() as stream:
                changed_rows = list(csv.DictReader(stream))
            self.assertEqual(len({r['flight_stream_sha256'] for r in changed_rows}), 1)
            self.assertNotEqual(changed_rows[0]['flight_stream_sha256'], rows[0]['flight_stream_sha256'])
            self.assertEqual(json.loads((changed / 'config.json').read_text())['environment'],
                             evaluated['environment'])


if __name__ == '__main__':
    unittest.main()
