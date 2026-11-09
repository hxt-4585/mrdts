import csv
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from methods.hmappo.curriculum import FlightCurriculum, MODES, scenario_settings
from methods.hmappo.rollout import Adapter, collect_slot
from methods.hmappo.run import train, load_checkpoint, evaluate
from methods.hmappo.support import build_scene, resolved_settings


class TestStagedTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_external_flight_does_not_sample_master(self):
        env = build_scene(resolved_settings(users=4, dag_nodes=3), episode=0)
        adapter = Adapter(env)
        master, members = adapter.policies(16, 'cpu')
        before = env.members.positions.copy()
        with patch.object(master, 'sample', side_effect=AssertionError('master sampled')):
            high, low, row = collect_slot(adapter, master, members,
                flight_actions=np.zeros((adapter.members, 2)), member_deterministic=True)
        self.assertEqual(high, [])
        np.testing.assert_array_equal(before, env.members.positions)
        self.assertEqual(sum(map(len, low.values())), row['member_decisions'])

    def test_curriculum_is_reproducible_and_flights_stay_in_bounds(self):
        base = resolved_settings(users=4, dag_nodes=3)
        seen = set()
        for episode in range(len(MODES) * 2):
            settings = scenario_settings(base, episode, (42, 43))
            env = build_scene(settings, episode=episode)
            flight = FlightCurriculum(env, episode, 45)
            seen.add((settings['env'].env_seed, flight.mode))
            clone = build_scene(settings, episode=episode)
            other = FlightCurriculum(clone, episode, 45)
            for _ in range(4):
                action = flight.actions(env)
                np.testing.assert_array_equal(action, other.actions(clone))
                self.assertTrue(np.all(np.linalg.norm(action, axis=1) <= 1.000001))
                self.assertFalse(env.members.apply_flight_actions(action, env.region.config.side_length).any())
                clone.members.apply_flight_actions(action, clone.region.config.side_length)
                env._refresh_topology()
                clone._refresh_topology()
        self.assertEqual(len(seen), len(MODES) * 2)

    def test_two_stages_freeze_correct_layer_and_evaluate(self):
        with tempfile.TemporaryDirectory(prefix='hmappo_staged_') as root:
            options = dict(episodes=2, slots=2, users=4, dag_nodes=3, hidden=16,
                           rollout_slots=2, ppo_epochs=1, seed=45, output_root=root)
            torch.manual_seed(45)
            env = build_scene(resolved_settings(users=4, dag_nodes=3), episode=0)
            initial_master, initial_members = Adapter(env).policies(16, 'cpu')
            low = train(**options, stage='member', scene_seeds=(42, 43))
            _, master, members, checkpoint = load_checkpoint(low / 'checkpoint.pt')
            self.assertTrue(all(torch.equal(v, initial_master.state_dict()[k])
                                for k, v in master.state_dict().items()))
            self.assertTrue(any(not torch.equal(v, initial_members.state_dict()[k])
                                for k, v in members.state_dict().items()))
            self.assertIsNone(checkpoint['master_optimizer'])
            high = train(**options, stage='master', member_checkpoint=low / 'checkpoint.pt')
            _, trained_master, frozen_members, high_checkpoint = load_checkpoint(high / 'checkpoint.pt')
            self.assertTrue(all(torch.equal(v, members.state_dict()[k])
                                for k, v in frozen_members.state_dict().items()))
            self.assertEqual(high_checkpoint['member_optimizers'], [])
            self.assertTrue(any(not torch.equal(v, initial_master.state_dict()[k])
                                for k, v in trained_master.state_dict().items()))
            evaluation = evaluate(low / 'checkpoint.pt', episodes=2, output_root=root,
                                  epoch_start=10000, scene_seeds=(1001, 1002))
            with (evaluation / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertTrue({int(r['scene_seed']) for r in rows}.issubset({1001, 1002}))
            self.assertTrue(all(int(r['epoch']) >= 10000 for r in rows))
