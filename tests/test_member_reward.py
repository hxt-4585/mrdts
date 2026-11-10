"""Member 时延奖励的手算与异步事件验证。"""

import unittest

import numpy as np

from env.contracts import PlacementDecision
from env.step.reward import MemberReward
from methods.scheduling import run_online_schedule
from tests.test_serial_runtime import make_runtime, request


class TestMemberReward(unittest.TestCase):
    def test_offload_saving_propagates_to_other_owned_dags(self):
        runtime, members, grounds, bs = make_runtime(slot_start=7.)
        runtime.servers[bs].core_frequencies = (30.,)
        first = request(members[0], grounds[0], cycles=(30.,))
        second = request(members[0], grounds[1], dag_id=1, cycles=(20.,))
        unrelated = request(members[1], grounds[2], cycles=(100.,))
        runtime.register_dags([first, second, unrelated])
        runtime.submit_placements({first.task_keys[0]: PlacementDecision(bs),
                                   unrelated.task_keys[0]: PlacementDecision(grounds[2])})
        runtime.advance_to_decision()
        reward = runtime.pop_member_rewards()[first.task_keys[0]]
        # 本地完成时延为 3、5 秒；实际分支为 1、3 秒。
        self.assertAlmostEqual(reward.compute_delay(), 2.)
        self.assertEqual(runtime.now, 8.)
        self.assertIsNone(runtime.trace(second.task_keys[0]).execution_node)
        self.assertIsNone(runtime.trace(unrelated.task_keys[0]).compute_finish_at)

    def test_remote_successor_transfer_can_outweigh_compute_saving(self):
        runtime, members, grounds, bs = make_runtime()
        runtime.servers[bs].core_frequencies = (30.,)
        dag = request(members[0], grounds[0], inputs=(0., 0.), cycles=(30., 10.),
                      edges=((0, 1),), sizes=(2.,))
        parent, child = runtime.register_dags([dag])
        runtime.submit_placements({parent: PlacementDecision(bs)})
        runtime.advance_to_decision()
        # 全本地为 4 秒；实际为 1 秒计算 + 两跳各 2 秒回传 + 1 秒计算。
        self.assertAlmostEqual(runtime.pop_member_rewards()[parent].compute_delay(), -2.)
        self.assertIsNone(runtime.trace(child).execution_node)

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
                               8. - (task.compute_finish_at - task.decision_at))
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

    def test_local_actions_have_zero_gain_with_multiple_owned_dags(self):
        runtime, members, grounds, _ = make_runtime(duration=1., slot_start=7.)
        a = request(members[0], grounds[0], inputs=(0., 0.), cycles=(10., 10.))
        b = request(members[0], grounds[1], dag_id=1)
        other = request(members[1], grounds[2], cycles=(100.,))
        runtime.register_dags([a, b, other])
        run_online_schedule(runtime, {members[0]: (*a.task_keys, *b.task_keys),
                                      members[1]: other.task_keys},
                            lambda key: PlacementDecision(grounds[key.user_id]))
        rewards = {key: reward.compute_delay() for key, reward in runtime.pop_member_rewards().items()}
        self.assertEqual([rewards[key] for key in (*a.task_keys, *b.task_keys)], [0., 0., 0.])
        self.assertEqual(rewards[other.task_keys[0]], 0.)
        self.assertEqual(runtime.trace(b.task_keys[0]).unfinished_dags_at_decision, 1)
        self.assertEqual(runtime.trace(b.task_keys[0]).unfinished_region_dags_at_decision, 2)
        self.assertEqual(runtime.trace(b.task_keys[0]).managed_dag_count, 2)
        self.assertEqual(runtime.pop_member_rewards(), {})
        runtime.finish_slot()
        self.assertEqual(runtime.pop_member_rewards(), {})

    def test_equal_speed_without_transfers_has_zero_gain_across_regions(self):
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
                         [0., 0., 0.])

    def test_other_member_completion_does_not_change_local_gain(self):
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
        self.assertEqual([rewards[key].compute_delay() for key in a.task_keys], [0., 0.])
        self.assertEqual(rewards[b.task_keys[0]].compute_delay(), 0.)

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
                         {fast.task_keys[0]: 0.})
        runtime.submit_placements({fast.task_keys[1]: PlacementDecision(bs)})
        runtime.advance_to_decision()
        rewards = {key: reward.compute_delay() for key, reward in runtime.pop_member_rewards().items()}
        self.assertEqual(rewards, {slow.task_keys[0]: 0., fast.task_keys[1]: -2.})
        self.assertEqual(runtime.trace(fast.task_keys[1]).compute_start_at, 3.)

        runtime, members, grounds, bs = make_runtime(duration=2.)
        dag = request(members[0], grounds[0], inputs=(2.,))
        runtime.register_dags([dag])
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(bs)})
        runtime.finish_slot()
        self.assertEqual(runtime.pop_member_rewards()[dag.task_keys[0]].compute_delay(), -4.)

    def test_completed_dags_remain_in_owner_average_denominator(self):
        runtime, members, grounds, bs = make_runtime(slot_start=7.)
        runtime.servers[bs].core_frequencies = (30.,)
        finished = request(members[0], grounds[0])
        current = request(members[0], grounds[1], dag_id=1, cycles=(30.,))
        runtime.register_dags([finished, current])
        runtime.submit_placements({finished.task_keys[0]: PlacementDecision(grounds[0])})
        runtime.advance_to_decision()
        runtime.pop_member_rewards()
        runtime.submit_placements({current.task_keys[0]: PlacementDecision(bs)})
        runtime.finish_slot()
        self.assertAlmostEqual(runtime.pop_member_rewards()[current.task_keys[0]].compute_delay(), 1.)

    def test_reference_order_respects_dependencies_with_reversed_node_ids(self):
        runtime, members, grounds, bs = make_runtime()
        runtime.servers[bs].core_frequencies = (30.,)
        dag = request(members[0], grounds[0], inputs=(0.,) * 3, cycles=(10., 10., 30.),
                      edges=((2, 1), (1, 0)), sizes=(0., 0.))
        runtime.register_dags([dag])
        runtime.submit_placements({dag.task_keys[2]: PlacementDecision(bs)})
        runtime.advance_to_decision()
        self.assertAlmostEqual(runtime.pop_member_rewards()[dag.task_keys[2]].compute_delay(), 2.)

    def test_local_baseline_preserves_ground_core_occupancy(self):
        runtime, members, grounds, bs = make_runtime()
        dag = request(members[0], grounds[0])
        local = request(None, grounds[0], dag_id=1, cycles=(30.,))
        runtime.register_dags([dag, local])
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(bs)})
        runtime.advance_to_decision()
        # 本地需等待已有 3 秒作业，再计算 1 秒；实际 BS 完成于 1 秒。
        self.assertAlmostEqual(runtime.pop_member_rewards()[dag.task_keys[0]].compute_delay(), 3.)

    def test_local_baseline_uses_available_core_frequency(self):
        runtime, members, grounds, bs = make_runtime(cores=2)
        runtime.servers[grounds[0]].core_frequencies = (10., 20.)
        dag = request(members[0], grounds[0], cycles=(20.,))
        local = request(None, grounds[0], dag_id=1, cycles=(30.,))
        runtime.register_dags([dag, local])
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(bs)})
        runtime.advance_to_decision()
        # 本地空闲的核心 1 只需 1 秒；BS 使用核心 0，需要 2 秒。
        self.assertAlmostEqual(runtime.pop_member_rewards()[dag.task_keys[0]].compute_delay(), -1.)

    def test_local_action_after_remote_parent_has_zero_gain(self):
        runtime, members, grounds, bs = make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 0.), cycles=(10., 10.),
                      edges=((0, 1),), sizes=(2.,))
        parent, child = runtime.register_dags([dag])
        runtime.submit_placements({parent: PlacementDecision(bs)})
        runtime.advance_to_decision()
        runtime.pop_member_rewards()
        runtime.submit_placements({child: PlacementDecision(grounds[0])})
        runtime.finish_slot()
        self.assertEqual(runtime.pop_member_rewards()[child].compute_delay(), 0.)

    def test_local_fifo_projection_preserves_simultaneous_idle_core_order(self):
        runtime, members, grounds, bs = make_runtime(cores=2)
        runtime.servers[grounds[0]].core_frequencies = (10., 20.)
        dag = request(members[0], grounds[0], cycles=(20.,))
        local = request(None, grounds[0], dag_id=1, inputs=(0.,) * 4,
                        cycles=(10., 20., 0., 20.))
        runtime.register_dags([dag, local])
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(bs)})
        runtime.advance_to_decision()
        # t=1 两个核心同时空闲，零任务先派给 0，20 cycles 派给 1。
        # 当前任务随后使用核心 0，完成于 3 秒；BS 分支完成于 2 秒。
        self.assertAlmostEqual(runtime.pop_member_rewards()[dag.task_keys[0]].compute_delay(), 1.)

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
