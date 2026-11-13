"""Member 观测固定使用调度初始队列，实际能耗仍正常结算。"""

import unittest
from dataclasses import replace

import numpy as np

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.runtime.task_runtime import TaskStatus
from tests import test_mrdts_env
from tests.test_serial_runtime import make_runtime, request


class TestMemberEnergyObservation(unittest.TestCase):
    def setUp(self):
        settings = test_mrdts_env.TestMRDTSEnv.settings()
        settings["uav"] = replace(settings["uav"], member_uav_count=4)
        self.env = MRDTSEnv(settings)
        self.env.generate_scene()
        self.env.members.region_ids[:] = 1
        self.env.members.energy_virtual_queue_j[:] = 100.
        self.env.user_member_ids = np.zeros(self.env.users.num_users, dtype=int)
        self.env._create_scheduling_runtime()
        self.runtime, self.members, self.grounds, self.bs = make_runtime(
            duration=4., on_member_energy=self.env._record_member_energy,
        )
        self.count = self.env.members.member_uav_count

    def observe_energy(self, key):
        observation = self.env._member_observation.build(key, self.runtime)
        return observation[4 + 4 * self.count:4 + 5 * self.count] * self.env.members.config.member_slot_energy

    def test_running_fifo_and_pending_transfers_do_not_change_initial_queue(self):
        dags = [request(self.members[i], self.grounds[i], inputs=(bits, 0.),
                        cycles=(cycles, 0.))
                for i, (bits, cycles) in enumerate(((0., 10.), (0., 20.), (10., 30.)))]
        self.runtime.register_dags(dags)
        self.runtime.submit_placements({dag.task_keys[0]: PlacementDecision(self.members[0]) for dag in dags})
        key = dags[0].task_keys[1]
        self.assertEqual(self.runtime.tasks[dags[0].task_keys[0]].status, TaskStatus.RUNNING)
        self.assertEqual(self.runtime.tasks[dags[1].task_keys[0]].status, TaskStatus.QUEUED_COMPUTE)
        self.assertIsNone(self.runtime.tasks[dags[2].task_keys[0]].input_arrival_at)
        before = self.env.members.energy_virtual_queue_j.copy()
        queued = tuple(item.task_key for item in self.runtime.servers[self.members[0]].queue)
        first = self.observe_energy(key)
        self.assertAlmostEqual(first[0], 100.)
        np.testing.assert_array_equal(self.observe_energy(key), first)
        np.testing.assert_array_equal(self.env.members.energy_virtual_queue_j, before)
        self.assertEqual(tuple(item.task_key for item in self.runtime.servers[self.members[0]].queue), queued)
        self.assertEqual(self.runtime.now, 0.)

    def test_waiting_running_and_completed_compute_keep_initial_queue(self):
        dag = request(self.members[0], self.grounds[0], inputs=(0., 0., 0.),
                      cycles=(0., 20., 0.), edges=((0, 1),), sizes=(1.,))
        self.runtime.register_dags([dag])
        self.runtime.submit_placements({dag.task_keys[0]: PlacementDecision(self.grounds[0])})
        self.runtime.submit_placements({dag.task_keys[1]: PlacementDecision(self.members[0])})
        task = self.runtime.tasks[dag.task_keys[1]]
        self.assertEqual(task.status, TaskStatus.WAITING_DATA)
        self.assertEqual(task.input_arrival_at, 0.)
        self.assertFalse(self.runtime.servers[self.members[0]].queue)
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[2])[0], 100.)
        self.runtime.advance_until(1.5)
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[2])[0], 100.)
        self.assertEqual(self.env.members.energy_virtual_queue_j[0], 100.)
        self.runtime.advance_until(3.)
        self.assertEqual(task.status, TaskStatus.FINISHED)
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[2])[0], 100.)
        self.assertEqual(self.env.members.energy_virtual_queue_j[0], 120.)

    def test_hidden_members_and_non_member_targets_do_not_add_energy(self):
        self.env.members.region_ids[1] = 2
        dags = [request(self.members[i], self.grounds[i], inputs=(0., 0.), cycles=(10., 0.))
                for i in range(3)]
        self.runtime.register_dags(dags)
        self.runtime.submit_placements({dag.task_keys[0]: PlacementDecision(target)
                                        for dag, target in zip(dags, (self.bs, self.members[1], self.grounds[2]))})
        energy = self.observe_energy(dags[0].task_keys[1])
        np.testing.assert_allclose(energy[:3], [100., 0., 100.])

    def test_initial_queue_survives_full_execution_settlement(self):
        dag = request(self.members[0], self.grounds[0], inputs=(0., 0.), cycles=(100., 0.))
        self.runtime.register_dags([dag])
        self.runtime.submit_placements({dag.task_keys[0]: PlacementDecision(self.members[0])})
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[1])[0], 100.)
        result = self.runtime.finish_slot()
        self.assertEqual(result.compute_energy_j, 100.)
        self.assertEqual(self.env.members.energy_virtual_queue_j[0], 200.)
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[1])[0], 100.)

    def test_snapshot_is_captured_before_first_observation_and_refreshed_next_window(self):
        dag = request(self.members[0], self.grounds[0])
        self.runtime.register_dags([dag])
        self.env.members.energy_virtual_queue_j[:] = 200.
        np.testing.assert_allclose(self.observe_energy(dag.task_keys[0]), 100.)
        self.env._create_scheduling_runtime()
        np.testing.assert_allclose(self.observe_energy(dag.task_keys[0]), 200.)

    def test_actual_transmit_energy_does_not_change_initial_queue_observation(self):
        dag = request(self.members[0], self.grounds[0], inputs=(1., 0.), cycles=(0., 0.))
        self.runtime.register_dags([dag])
        self.runtime.submit_placements({dag.task_keys[0]: PlacementDecision(self.members[1])})
        self.runtime.advance_to_decision()
        self.assertGreater(self.env.members.energy_virtual_queue_j[0], 100.)
        np.testing.assert_allclose(self.observe_energy(dag.task_keys[1]), 100.)


if __name__ == '__main__':
    unittest.main()
