"""Real default-size integration checks for the additive RL adapter."""

import importlib.util
import unittest

import numpy as np

from env.runtime.slot_result import DAGResult, SlotResult


class TestRLAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('methods.rl_baseline') is None:
            return
        from methods.rl_baseline.adapter import DelayEnvironment
        cls.env = DelayEnvironment()

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('methods.rl_baseline'),
                             'The additive RL adapter must exist')

    def test_reward_includes_failed_dags_and_uses_relative_time(self):
        from methods.rl_baseline.adapter import delay_metrics
        result = SlotResult(8., 9., (DAGResult((0, 0, 0), 8.2, False),
                                    DAGResult((1, 1, 0), None, True)), 0., 0.)
        metrics = delay_metrics(result)
        self.assertAlmostEqual(metrics['mean_delay_s'], .6)
        self.assertAlmostEqual(metrics['reward'], -.6)
        self.assertAlmostEqual(metrics['failure_rate'], .5)

    def test_default_batch_is_atomic_and_ground_delay_is_exact(self):
        self.env.reset()
        batch = self.env.begin()
        self.assertEqual(len(batch.requests), 100)
        self.assertEqual(len(batch.plan.order), 1000)
        self.assertEqual(self.env.member_count, 12)
        self.assertEqual(self.env.master_count, 4)
        self.assertFalse(batch.runtime.tasks)
        expected = np.mean([r.dag.node_features[:, 1].sum() /
                            self.env.users.config.core_frequency for r in batch.requests])
        actions = {key: 0 for key in batch.plan.order}
        metrics = self.env.finish(batch, actions)
        self.assertAlmostEqual(metrics['mean_delay_s'], expected, places=10)
        self.assertEqual(metrics['failure_rate'], 0.)
        self.assertIsNone(self.env.simulator.runtime)

    def test_cross_region_refreshes_ownership_and_candidate_mask(self):
        from methods.rl_baseline.observations import MemberPlanning
        self.env.reset()
        members = self.env.members
        source = int(members.region_ids[0])
        target = next(r for r in range(1, 5) if r != source)
        target_id = int(np.flatnonzero(members.region_ids[:12] == target)[0])
        # Controlled valid migration state; identity remains 0 across the boundary.
        members.positions[0, :2] = members.positions[target_id, :2] + [1., 0.]
        batch = self.env.begin()
        self.assertEqual(int(members.region_ids[0]), target)
        planning = MemberPlanning(self.env, batch)
        for owner, order in batch.plan.member_orders.items():
            observation, mask = planning.observe(order[0])
            self.assertEqual(mask.shape, (14,))
            self.assertTrue(mask[0] and mask[-1])
            region = int(members.region_ids[owner.index])
            self.assertEqual(bool(mask[1]), region == target)
            self.assertTrue(np.isfinite(observation).all())
        self.env.finish(batch, {key: 0 for key in batch.plan.order})

    def test_empty_occupied_region_rejects_joint_flight(self):
        self.env.reset()
        # Put every Member of one region on a cell adjacent to another region,
        # then ask all of them to cross. Rejection must preserve ALL positions.
        region_map = self.env.region.region_map
        cell = self.env.region.config.cell_size
        boundary = np.argwhere(region_map[:, :-1] != region_map[:, 1:])[0]
        row, col = map(int, boundary)
        source = int(region_map[row, col])
        ids = np.flatnonzero(self.env.members.region_ids[:12] == source)
        self.env.members.positions[ids, :2] = [(col + 1) * cell - 1., (row + .5) * cell]
        self.env.members.positions[ids, 1] += np.arange(len(ids)) * .1
        before = self.env.members.positions.copy()
        actions = np.zeros((12, 2), dtype=np.float32)
        actions[ids, 0] = 1.
        batch = self.env.begin(actions)
        self.assertTrue(batch.flight_rejected)
        np.testing.assert_array_equal(self.env.members.positions, before)
        self.env.finish(batch, {key: 0 for key in batch.plan.order})

    def test_actual_flight_can_cross_region_without_ending_agent_identity(self):
        self.env.reset()
        region_map = self.env.region.region_map
        row, col = map(int, np.argwhere(region_map[:, :-1] != region_map[:, 1:])[0])
        source, target = int(region_map[row, col]), int(region_map[row, col + 1])
        member = int(np.flatnonzero(self.env.members.region_ids[:12] == source)[0])
        cell = self.env.region.config.cell_size
        self.env.members.positions[member, :2] = [(col + 1) * cell - 1., (row + .5) * cell]
        actions = np.zeros((12, 2), dtype=np.float32)
        actions[member, 0] = 1.
        batch = self.env.begin(actions)
        self.assertFalse(batch.flight_rejected)
        self.assertEqual(batch.migrations, 1)
        self.assertEqual(int(self.env.members.region_ids[member]), target)
        self.env.finish(batch, {key: 0 for key in batch.plan.order})

    def test_other_members_private_plan_does_not_change_local_actor_input(self):
        from methods.rl_baseline.observations import MemberPlanning
        self.env.reset()
        batch = self.env.begin()
        planning = MemberPlanning(self.env, batch)
        orders = list(batch.plan.member_orders.values())
        key, other = orders[0][0], orders[1][0]
        before, mask = planning.observe(key)
        planning.assign(other, other.owner_member_id + 1)
        after, later_mask = planning.observe(key)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(mask, later_mask)
        self.assertFalse(batch.runtime.tasks)
        self.env.finish(batch, {key: 0 for key in batch.plan.order})

    def test_workload_offset_reserves_disjoint_stream_without_changing_config(self):
        from methods.rl_baseline.adapter import DelayEnvironment
        self.env.reset()
        other = DelayEnvironment(workload_offset=1)
        reference = DelayEnvironment()
        skipped = reference.begin()
        reference.finish(skipped, {key: 0 for key in skipped.plan.order})
        a, b = reference.begin(), other.begin()
        for x, y in zip(a.requests, b.requests):
            np.testing.assert_array_equal(x.dag.node_features, y.dag.node_features)
            self.assertEqual(x.dag.edges, y.dag.edges)
        self.assertEqual(other.generator.config.seed, 20)
        reference.finish(a, {key: 0 for key in a.plan.order})
        other.finish(b, {key: 0 for key in b.plan.order})


if __name__ == '__main__':
    unittest.main()
