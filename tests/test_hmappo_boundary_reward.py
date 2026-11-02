from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from methods.hmappo.rollout import collect_slot


class TestHmappoBoundaryReward(unittest.TestCase):
    def collect(self, violations, **options):
        # 飞行前 Master 0 管理成员 0、1，Master 1 管理成员 2。
        samples = [SimpleNamespace(action=np.zeros(6), mask=np.repeat(mask, 2), reward=0.)
                   for mask in ([True, True, False], [False, False, True])]
        env = Mock(epoch_index=0, step_index=1)
        env.members.region_ids = np.array([1, 1, 2])
        env.members.positions = np.zeros((3, 3))
        env.masters.region_ids = np.array([1, 2])

        def begin_step(actions):
            # 模拟飞行后拓扑改变，惩罚仍须归给产生动作的 Master。
            env.members.region_ids[:] = [2, 2, 1]
            return SimpleNamespace(requests=(), ready_member_ids=(),
                                   flight_violations=np.array(violations))

        env.begin_step.side_effect = begin_step
        env.end_step.return_value = SimpleNamespace(
            member_rewards={}, master_reward=-.2, slot_result=None,
            total_energy_j=np.zeros(3), energy_virtual_queue_j=np.zeros(3))
        adapter = Mock(env=env, members=3)
        with patch('methods.hmappo.rollout.decide', return_value=samples), \
                patch('methods.hmappo.rollout.ERS') as ers, \
                patch('methods.hmappo.rollout.slot_metrics', return_value={}):
            ers.return_value.plan.return_value.member_orders = {}
            return collect_slot(adapter, None, None, **options)

    def test_default_penalty_counts_only_preflight_owned_violations(self):
        high, _, metrics = self.collect([True, True, False])
        np.testing.assert_allclose([sample.reward for sample in high], [-.4, -.2])
        self.assertEqual(metrics['master_reward'], -.2)
        self.assertAlmostEqual(metrics['master_training_reward_mean'], -.3)
        self.assertAlmostEqual(metrics['master_boundary_penalty_sum'], .2)

    def test_no_violation_preserves_delay_reward(self):
        high, _, _ = self.collect([False, False, False])
        np.testing.assert_allclose([sample.reward for sample in high], [-.2, -.2])

    def test_configured_coefficient_and_zero_disable(self):
        for coefficient in (0., .25):
            with self.subTest(coefficient=coefficient):
                high, _, _ = self.collect([False, True, True], boundary_penalty=coefficient)
                np.testing.assert_allclose([sample.reward for sample in high],
                                           [-.2 - coefficient, -.2 - coefficient])
