"""整窗静态调度的逐节点时延奖励与能耗归属。"""

import unittest

import numpy as np

from env.contracts import PlacementDecision
from env.step.reward import MemberReward
from tests.test_serial_runtime import make_runtime, request


class TestMemberReward(unittest.TestCase):
    def test_energy_is_attributed_to_input_relay_predecessor_and_executor(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 2.), cycles=(10., 20.),
                      edges=((0, 1),), sizes=(3.,))
        parent, child = runtime.register_dags([dag])
        runtime.submit_placements({parent: PlacementDecision(members[1]), child: PlacementDecision(members[2])})
        slot = runtime.finish_slot()
        results = runtime.pop_member_rewards()
        reward = results[child]
        self.assertIsInstance(reward, MemberReward)
        task = runtime.trace(child)
        expected = [task.input_record.tx_energy_j,
                    task.predecessor_records[parent].tx_energy_j, 20., 0.]
        np.testing.assert_allclose(reward.compute_energy(), expected)
        self.assertTrue(np.all(reward.compute_energy()[:3] > 0.))
        self.assertAlmostEqual(reward.compute_delay(),
                               -(task.compute_finish_at - runtime.trace(parent).compute_finish_at) / 30.)
        np.testing.assert_allclose(results[parent].compute_energy(), [0., 10., 0., 0.])
        np.testing.assert_allclose(
            sum(result.compute_energy() for result in results.values()),
            [row.compute_energy_j + row.tx_energy_j for row in slot.member_energy],
        )
        self.assertEqual(runtime.pop_member_rewards(), {})

    def test_relay_to_ground_and_bs_transmissions_keep_member_only_energy(self):
        for source in ('member', 'bs'):
            with self.subTest(source=source):
                runtime, members, grounds, bs = make_runtime()
                dag = request(members[0], grounds[0], inputs=(0., 0.), cycles=(10., 10.),
                              edges=((0, 1),), sizes=(2.,))
                parent, child = runtime.register_dags([dag])
                target = members[1] if source == 'member' else bs
                runtime.submit_placements({parent: PlacementDecision(target), child: PlacementDecision(grounds[0])})
                runtime.finish_slot()
                results = runtime.pop_member_rewards()
                np.testing.assert_allclose(results[child].compute_energy(),
                                           [2., 2. if source == 'member' else 0., 0., 0.])

    def test_rewards_follow_placement_order_across_dags_of_each_owner(self):
        runtime, members, grounds, _ = make_runtime(duration=2., slot_start=7.)
        dag = request(members[0], grounds[0], inputs=(0., 0., 0.),
                      cycles=(10., 30., 10.), edges=((0, 2), (1, 2)), sizes=(4., 0.))
        other = request(members[1], grounds[1])
        local = request(None, grounds[2])
        same_owner = request(members[0], grounds[3], cycles=(20.,))
        runtime.register_dags([dag, other, local, same_owner])
        a, b, child = dag.task_keys
        other_root, = other.task_keys
        sibling, = same_owner.task_keys
        runtime.submit_placements({b: PlacementDecision(members[0]), other_root: PlacementDecision(grounds[1]),
                                   sibling: PlacementDecision(members[1]), a: PlacementDecision(grounds[0]),
                                   child: PlacementDecision(members[0])})
        runtime.advance_until(10.)
        self.assertEqual(runtime.pop_member_rewards(), {})
        runtime.finish_slot()
        rewards = runtime.pop_member_rewards()
        # Owner 0 submits B, sibling, A, child. They finish at 10, 9, 8, 13 (slot starts at 7).
        self.assertEqual({key: reward.compute_delay() for key, reward in rewards.items()},
                         {b: -1.5, other_root: -.5, sibling: 0., a: 0., child: -1.5})
        self.assertGreater(rewards[sibling].compute_energy()[1], 0.)
        runtime.finish_slot()
        self.assertEqual(runtime.pop_member_rewards(), {})

    def test_equal_finish_charges_only_the_first_submitted_node(self):
        runtime, members, grounds, _ = make_runtime(duration=2., cores=4)
        dag = request(members[0], grounds[0], inputs=(0.,) * 4, cycles=(10., 10., 10., 20.),
                      edges=((0, 2), (1, 2)), sizes=(0., 0.))
        keys = runtime.register_dags([dag])
        runtime.submit_placements({k: PlacementDecision(members[0]) for k in reversed(keys)})
        runtime.finish_slot()
        rewards = runtime.pop_member_rewards()
        self.assertEqual([rewards[k].compute_delay() for k in keys], [0., 0., 0., -1.])

    def test_input_and_compute_queue_wait_are_charged_to_root(self):
        runtime, members, grounds, _ = make_runtime(duration=2.)
        first = request(members[1], grounds[1], cycles=(30.,))
        second = request(members[0], grounds[0], inputs=(1.,))
        runtime.register_dags([first, second])
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(members[0]) for dag in (first, second)})
        runtime.finish_slot()
        rewards = runtime.pop_member_rewards()
        self.assertEqual(rewards[first.task_keys[0]].compute_delay(), -1.5)
        self.assertEqual(rewards[second.task_keys[0]].compute_delay(), -2.)

    def test_zero_duration_node_settles_at_slot_end(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], cycles=(0.,))
        key, = runtime.register_dags([dag])
        runtime.submit_placements({key: PlacementDecision(grounds[0])})
        self.assertEqual(runtime.pop_member_rewards(), {})
        runtime.finish_slot()
        self.assertEqual(runtime.pop_member_rewards()[key].compute_delay(), 0.)

if __name__ == '__main__':
    unittest.main()
