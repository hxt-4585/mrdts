import unittest

import numpy as np
import torch

from methods.hmappo.ers import ERS
from methods.hmappo.rollout import Adapter, decide
from methods.hmappo.support import build_scene, resolved_settings


class TestHmappoObservation(unittest.TestCase):
    def setUp(self):
        self.env = build_scene(resolved_settings(users=4, dag_nodes=3), episode=0)
        self.adapter = Adapter(self.env)

    def test_master_inputs_ignore_energy_and_preserve_positions(self):
        env, adapter = self.env, self.adapter
        before = adapter.master_inputs()
        env.members.energy_virtual_queue_j[:] = 12345.
        after = adapter.master_inputs()
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
        observations, states, _ = after
        self.assertEqual(observations.shape, (adapter.masters, 3 * adapter.members + adapter.masters))
        self.assertEqual(states.shape, (adapter.masters, adapter.master_state_dim))
        self.assertEqual(adapter.global_dim, 2 * adapter.members + adapter.members * adapter.masters + 1)
        for i in range(adapter.masters):
            expected = env.observe_master(i).reshape(-1, 3)[:, :2].ravel()
            np.testing.assert_array_equal(observations[i, :2 * adapter.members], expected.astype(np.float32))
        master, _ = adapter.policies(16, 'cpu')
        with torch.no_grad():
            self.assertEqual(len(decide(master, after, True)), adapter.masters)

    def test_member_inputs_ignore_energy_and_preserve_other_features(self):
        env, adapter = self.env, self.adapter
        begin = env.begin_step(np.zeros((adapter.members, 2)))
        plan = ERS(env._runtime).plan(begin.requests)
        keys = [plan.member_orders[adapter.nodes[owner]][0] for owner in begin.ready_member_ids]
        self.assertTrue(keys)
        before = adapter.member_inputs(keys)
        env.members.energy_virtual_queue_j[:] += 12345.
        after = adapter.member_inputs(keys)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
        observations, states, _ = after
        self.assertEqual(observations.shape, (len(keys), 5 * adapter.members + 8))
        self.assertIs(states, observations)
        self.assertEqual(states.shape, (len(keys), adapter.member_state_dim))
        for key, observation in zip(keys, observations):
            raw = env.observe_member(key)
            expected = np.concatenate((raw[:4 + 4 * adapter.members], raw[4 + 5 * adapter.members:]))
            np.testing.assert_array_equal(observation, expected.astype(np.float32))
        _, members = adapter.policies(16, 'cpu')
        with torch.no_grad():
            for row, key in enumerate(keys):
                inputs = tuple(values[row:row + 1] for values in after)
                self.assertEqual(len(decide(members[key.owner_member_id], inputs, True)), 1)
