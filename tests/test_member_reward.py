"""Member 时延奖励的手算与异步事件验证。"""

import unittest

import numpy as np

from env.contracts import PlacementDecision
from env.step.reward import MemberReward
from methods.scheduling import run_online_schedule
from tests.test_serial_runtime import make_runtime, request


class TestMemberReward(unittest.TestCase):
    def test_energy_is_attributed_to_input_relay_predecessor_and_executor(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 2.), cycles=(10., 20.),
                      edges=((0, 1),), sizes=(3.,))
        parent, child = runtime.register_dags([dag])
        runtime.submit_placements({parent: PlacementDecision(members[1])})
        runtime.advance_to_decision()
        runtime.submit_placements({child: PlacementDecision(members[2])})
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
                               -(task.compute_finish_at - task.decision_at) / 30.)
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
                runtime.submit_placements({parent: PlacementDecision(target)})
                runtime.advance_to_decision()
                runtime.submit_placements({child: PlacementDecision(grounds[0])})
                runtime.finish_slot()
                results = runtime.pop_member_rewards()
                np.testing.assert_allclose(results[child].compute_energy(),
                                           [2., 2. if source == 'member' else 0., 0., 0.])

    def test_delay_weights_all_unfinished_dags_in_owner_region_at_decision(self):
        runtime, members, grounds, _ = make_runtime(duration=1., slot_start=7.)
        a = request(members[0], grounds[0], inputs=(0., 0.), cycles=(10., 10.))
        b = request(members[0], grounds[1], dag_id=1)
        other = request(members[1], grounds[2], cycles=(100.,))
        runtime.register_dags([a, b, other])
        run_online_schedule(runtime, {members[0]: (*a.task_keys, *b.task_keys),
                                      members[1]: other.task_keys},
                            lambda key: PlacementDecision(grounds[key.user_id]))
        rewards = {key: reward.compute_delay() for key, reward in runtime.pop_member_rewards().items()}
        self.assertEqual([rewards[key] for key in (*a.task_keys, *b.task_keys)], [-1., -1., -2/3])
        self.assertEqual(rewards[other.task_keys[0]], -10.)
        self.assertEqual(runtime.trace(b.task_keys[0]).unfinished_dags_at_decision, 1)
        self.assertEqual(runtime.trace(b.task_keys[0]).unfinished_region_dags_at_decision, 2)
        self.assertEqual(runtime.trace(b.task_keys[0]).managed_dag_count, 2)
        self.assertEqual(runtime.pop_member_rewards(), {})
        runtime.finish_slot()
        self.assertEqual(runtime.pop_member_rewards(), {})

    def test_delay_excludes_other_regions_and_uses_owner_not_executor_region(self):
        runtime, members, grounds, _ = make_runtime(duration=1.)
        runtime.update_topology(
            dict(zip(members, (1, 1, 2, 2))), dict(zip(grounds, members)))
        dags = [request(member, ground) for member, ground in zip(members[:3], grounds[:3])]
        runtime.register_dags(dags)
        runtime.submit_placements({
            dags[0].task_keys[0]: PlacementDecision(members[2]),
            dags[1].task_keys[0]: PlacementDecision(grounds[1]),
            dags[2].task_keys[0]: PlacementDecision(grounds[2]),
        })
        runtime.finish_slot()
        rewards = runtime.pop_member_rewards()
        self.assertEqual([rewards[dag.task_keys[0]].compute_delay() for dag in dags],
                         [-2/3, -2/3, -1/3])

    def test_region_weight_decreases_when_another_member_finishes_its_dag(self):
        runtime, members, grounds, _ = make_runtime(duration=1.)
        a = request(members[0], grounds[0], inputs=(0., 0.), cycles=(10., 10.))
        b = request(members[1], grounds[1])
        runtime.register_dags([a, b])
        runtime.submit_placements({a.task_keys[0]: PlacementDecision(grounds[0]),
                                   b.task_keys[0]: PlacementDecision(grounds[1])})
        runtime.advance_to_decision()
        runtime.submit_placements({a.task_keys[1]: PlacementDecision(grounds[0])})
        runtime.finish_slot()
        rewards = runtime.pop_member_rewards()
        self.assertEqual([rewards[key].compute_delay() for key in a.task_keys], [-1., -.5])
        self.assertEqual(rewards[b.task_keys[0]].compute_delay(), -1.)

    def test_rewards_wait_for_actual_finish_and_include_queue_and_transfer(self):
        runtime, members, grounds, bs = make_runtime(duration=2.)
        slow = request(members[0], grounds[0], cycles=(30.,))
        fast = request(members[1], grounds[1], inputs=(0., 0.), cycles=(10., 10.))
        runtime.register_dags([slow, fast])
        runtime.submit_placements({slow.task_keys[0]: PlacementDecision(bs),
                                   fast.task_keys[0]: PlacementDecision(grounds[1])})
        self.assertEqual(runtime.pop_member_rewards(), {})
        runtime.advance_to_decision()
        self.assertEqual({key: reward.compute_delay() for key, reward in runtime.pop_member_rewards().items()},
                         {fast.task_keys[0]: -.5})
        runtime.submit_placements({fast.task_keys[1]: PlacementDecision(bs)})
        runtime.advance_to_decision()
        rewards = {key: reward.compute_delay() for key, reward in runtime.pop_member_rewards().items()}
        self.assertEqual(rewards, {slow.task_keys[0]: -1.5, fast.task_keys[1]: -1.5})
        self.assertEqual(runtime.trace(fast.task_keys[1]).compute_start_at, 3.)

        runtime, members, grounds, bs = make_runtime(duration=2.)
        dag = request(members[0], grounds[0], inputs=(2.,))
        runtime.register_dags([dag])
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(bs)})
        runtime.finish_slot()
        self.assertEqual(runtime.pop_member_rewards()[dag.task_keys[0]].compute_delay(), -2.5)

    def test_zero_duration_and_members_without_dags(self):
        runtime, members, grounds, _ = make_runtime()
        self.assertEqual(runtime.pop_member_rewards(), {})
        dag = request(members[0], grounds[0], cycles=(0.,))
        runtime.register_dags([dag])
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(grounds[0])})
        self.assertEqual(runtime.pop_member_rewards()[dag.task_keys[0]].compute_delay(), 0.)
        self.assertEqual(runtime.pop_member_rewards(), {})


if __name__ == '__main__':
    unittest.main()
