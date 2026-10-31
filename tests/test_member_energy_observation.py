"""Member 观测中的已到达计算负载与实际能耗结算。"""

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
        self.runtime, self.members, self.grounds, self.bs = make_runtime(
            duration=4., on_member_energy=self.env._record_member_energy,
        )
        self.count = self.env.members.member_uav_count

    def observe_energy(self, key):
        observation = self.env._member_observation.build(key, self.runtime)
        return observation[4 + 4 * self.count:4 + 5 * self.count] * self.env.members.config.member_slot_energy

    def test_running_and_fifo_energy_is_read_only_and_transfers_are_excluded(self):
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
        queued = self.runtime.servers[self.members[0]].queued_task_keys
        first = self.observe_energy(key)
        self.assertAlmostEqual(first[0], 130.)  # kappa=.01, f=10 -> 1 J/cycle
        np.testing.assert_array_equal(self.observe_energy(key), first)
        np.testing.assert_array_equal(self.env.members.energy_virtual_queue_j, before)
        self.assertEqual(self.runtime.servers[self.members[0]].queued_task_keys, queued)
        self.assertEqual(self.runtime.now, 0.)

    def test_input_arrived_waiting_for_predecessor_is_counted_then_not_double_counted(self):
        dag = request(self.members[0], self.grounds[0], inputs=(0., 0., 0.),
                      cycles=(0., 20., 0.), edges=((0, 1),), sizes=(1.,))
        self.runtime.register_dags([dag])
        self.runtime.submit_placements({dag.task_keys[0]: PlacementDecision(self.grounds[0])})
        self.runtime.submit_placements({dag.task_keys[1]: PlacementDecision(self.members[0])})
        task = self.runtime.tasks[dag.task_keys[1]]
        self.assertEqual(task.status, TaskStatus.WAITING_DATA)
        self.assertEqual(task.input_arrival_at, 0.)
        self.assertFalse(self.runtime.servers[self.members[0]].queue)
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[2])[0], 120.)
        self.runtime.advance_until(1.5)
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[2])[0], 120.)
        self.assertEqual(self.env.members.energy_virtual_queue_j[0], 100.)
        self.runtime.advance_until(3.)
        self.assertEqual(task.status, TaskStatus.FINISHED)
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[2])[0], 120.)
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

    def test_full_prediction_does_not_change_partial_deadline_settlement(self):
        dag = request(self.members[0], self.grounds[0], inputs=(0., 0.), cycles=(100., 0.))
        self.runtime.register_dags([dag])
        self.runtime.submit_placements({dag.task_keys[0]: PlacementDecision(self.members[0])})
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[1])[0], 200.)
        result = self.runtime.finish_slot()
        self.assertEqual(result.compute_energy_j, 40.)
        self.assertEqual(self.env.members.energy_virtual_queue_j[0], 140.)
        self.assertAlmostEqual(self.observe_energy(dag.task_keys[1])[0], 140.)


if __name__ == '__main__':
    unittest.main()
