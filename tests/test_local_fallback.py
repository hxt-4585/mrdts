"""无 Member 区域的自动本地执行与正常卸载共用同一时钟。"""

import unittest
import numpy as np

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.runtime.task_runtime import TaskStatus
from methods.random.run import RandomStreams, resolved_settings, run_slot
from tests.test_serial_runtime import make_runtime, request


class TestLocalFallback(unittest.TestCase):
    def test_local_fork_join_uses_ground_core_and_finishes_without_decisions(self):
        runtime, members, grounds, bs = make_runtime(duration=1.)
        local = request(None, grounds[0], inputs=(100.,) * 4,
                        cycles=(10., 20., 30., 40.),
                        edges=((0, 1), (0, 2), (1, 3), (2, 3)), sizes=(100.,) * 4)
        runtime.register_dags([local])
        self.assertEqual(runtime.ready_member_ids, ())
        result = runtime.finish_slot()
        traces = [runtime.trace(key) for key in local.task_keys]
        self.assertEqual([t.compute_start_at for t in traces], [0., 1., 3., 6.])
        self.assertEqual(result.dags[0].completion_time, 10.)
        self.assertEqual(result.failed_dags, (local.key,))
        self.assertEqual(result.slot_end, 10.)
        self.assertTrue(all(t.execution_node == grounds[0] for t in traces))
        self.assertFalse(runtime.transfer_jobs)
        self.assertFalse(runtime.pop_member_rewards())
        self.assertEqual(result.compute_energy_j, 0.)
        self.assertEqual(result.tx_energy_j, 0.)

    def test_local_execution_does_not_block_member_decisions_or_earn_member_rewards(self):
        runtime, members, grounds, bs = make_runtime(duration=1.)
        local = request(None, grounds[0], cycles=(100.,))
        managed = request(members[1], grounds[1], inputs=(0., 0.), cycles=(10., 10.))
        runtime.register_dags([local, managed])
        self.assertEqual(runtime.ready_member_ids, (1,))
        runtime.submit_placements({managed.task_keys[0]: PlacementDecision(bs)})
        self.assertEqual(runtime.advance_to_decision(), (1,))
        self.assertEqual(runtime.now, 1.)
        runtime.submit_placements({managed.task_keys[1]: PlacementDecision(bs)})
        self.assertEqual(runtime.advance_to_decision(), ())
        result = runtime.finish_slot()
        self.assertFalse(result.finished_dags)
        self.assertEqual(result.failed_dags, (local.key, managed.key))
        self.assertEqual(result.slot_end, 10.)
        self.assertEqual(set(runtime.pop_member_rewards()), set(managed.task_keys))

    def test_zero_duration_local_tasks_and_independent_ground_devices(self):
        runtime, members, grounds, bs = make_runtime(duration=1.)
        dags = [request(None, g, inputs=(0., 0., 0.), cycles=(0., 0., 10.),
                        edges=((0, 2), (1, 2)), sizes=(0., 0.)) for g in grounds]
        runtime.register_dags(dags)
        result = runtime.finish_slot()
        self.assertEqual(len(result.finished_dags), 4)
        self.assertTrue(all(d.completion_time == 1. for d in result.dags))
        self.assertTrue(all(t.status is TaskStatus.FINISHED for t in runtime.tasks.values()))

    def make_env(self):
        settings = resolved_settings(users=8, dag_nodes=4)
        environment = MRDTSEnv(settings)
        environment.generate_scene()
        environment.reset(epoch_index=0)
        # 将所有 Member 移入区域 1，其他区域保留原用户。
        environment.members.positions[:-1, :2] = environment.masters.positions[0, :2]
        return environment

    def test_empty_regions_complete_in_random_baseline(self):
        environment = self.make_env()
        result, _ = run_slot(environment, RandomStreams())
        self.assertEqual(len(result.finished_dags), environment.users.num_users)
        local_users = set(np.flatnonzero(environment.user_member_ids == -1))
        self.assertTrue(local_users)
        local_results = [d for d in result.dags if d.key[1] in local_users]
        self.assertEqual(len(local_results), len(local_users))
        self.assertTrue(all(d.key[0] == -1 for d in local_results))

    def test_all_users_local_settle_reward_hover_and_next_slot_reassociation(self):
        environment = self.make_env()
        # 本窗口所有用户放在没有 Member 的区域 2。
        environment.users.positions[:, :2] = environment.masters.positions[1, :2]
        environment.users.region_ids[:] = 2
        actions = np.zeros((environment.members.member_uav_count, 2))
        begin = environment.begin_step(actions)
        self.assertEqual(begin.ready_member_ids, ())
        self.assertTrue(all(r.owner_member is None for r in begin.requests))
        result = environment.end_step()
        expected_delays = [r.dag.node_features[:, 1].sum() / environment.users.config.core_frequency
                           for r in begin.requests]
        np.testing.assert_allclose([d.completion_time for d in result.slot_result.dags], expected_delays)
        self.assertAlmostEqual(result.master_reward, -np.mean(expected_delays) / result.scheduling_duration_s)
        self.assertEqual(result.member_rewards, {})
        np.testing.assert_allclose(result.compute_energy_j, 0.)
        np.testing.assert_allclose(result.tx_energy_j, 0.)
        np.testing.assert_allclose(result.hover_energy_j,
                                   environment._hover_power_w * result.slot_result.slot_end)
        np.testing.assert_allclose(result.energy_virtual_queue_j,
                                   np.maximum(result.total_energy_j - environment.members.config.member_slot_energy, 0.))
        environment.members.positions[:-1] = environment._initial_member_positions[:-1]
        following = environment.begin_step(actions)
        self.assertTrue(following.ready_member_ids)
        self.assertTrue(all(r.owner_member is not None for r in following.requests))


if __name__ == '__main__':
    unittest.main()
