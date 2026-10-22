"""飞行、事件队列更新和跨窗口结算的环境集成测试。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.types import EntityKind, EntityRef
from tests import test_mrdts_env
from tests.test_serial_runtime import request


class TestEndStep(unittest.TestCase):
    def make_environment(self, *, duration=2., budget=145.):
        settings = test_mrdts_env.TestMRDTSEnv.settings()
        settings["user"] = replace(settings["user"], total_users=8)
        settings["env"] = replace(settings["env"], scheduling_duration_s=duration, steps_per_epoch=2)
        settings["uav"] = replace(settings["uav"], member_slot_energy=budget)
        environment = MRDTSEnv(settings)
        environment.generate_scene()
        environment.reset(epoch_index=0)
        return environment

    def test_flight_formula_rejected_motion_budget_and_independent_results(self):
        env = self.make_environment()
        members = env.members
        count = members.member_uav_count
        members.positions[0, :2] = env.region.config.side_length / 2.
        members.positions[1, 0] = .1
        previous = members.positions[:count, :2].astype(float)
        members.energy_virtual_queue_j[:] = 20.
        actions = np.zeros((count, 2))
        actions[0] = [.6, -.8]
        actions[1] = [-1., 0.]
        begin = env.begin_step(actions)
        self.assertTrue(begin.flight_violations[1])
        p = members.config.member_propulsion
        duration = env.env_config.flight_duration_s
        speed = np.linalg.norm((members.positions[:count, :2] - previous) / duration, axis=1)
        expected_flight = (p.u1 * (1. + 3. * speed**2 / p.tip_speed**2) + p.u2 * speed**3
                           + p.u3 * np.sqrt(np.sqrt(p.u4 + speed**4 / 4.) - speed**2 / 2.)) * duration
        hover_power = p.u1 + p.u3 * p.u4**.25
        np.testing.assert_allclose(members.energy_virtual_queue_j, 20. + expected_flight)
        np.testing.assert_allclose(env.observe_master(0).reshape(count, 3)[0, 2],
                                   members.energy_virtual_queue_j[0] / members.config.member_slot_energy)
        runtime = env._runtime
        result = env.end_step()
        self.assertEqual(len(result.slot_result.failed_dags), env.users.num_users)
        np.testing.assert_allclose(result.flight_energy_j, expected_flight)
        np.testing.assert_allclose(result.hover_energy_j, hover_power * 2.)
        np.testing.assert_array_equal(result.compute_energy_j, 0.)
        np.testing.assert_array_equal(result.tx_energy_j, 0.)
        np.testing.assert_allclose(result.total_energy_j, expected_flight + hover_power * 2.)
        expected_queue = np.maximum(0., 20. + result.total_energy_j - 145.)
        np.testing.assert_allclose(result.energy_virtual_queue_j, expected_queue)
        np.testing.assert_allclose(members.energy_virtual_queue_j, expected_queue)
        self.assertFalse(np.shares_memory(result.energy_virtual_queue_j, members.energy_virtual_queue_j))
        self.assertEqual(env.step_index, 1)
        self.assertEqual(env._next_slot_start, 2.)
        self.assertFalse(result.epoch_done)
        self.assertIsNone(env._runtime)
        before = members.energy_virtual_queue_j.copy()
        self.assertIs(runtime.finish_slot(), result.slot_result)
        np.testing.assert_array_equal(members.energy_virtual_queue_j, before)
        env.begin_step(np.zeros_like(actions))
        self.assertEqual(env._runtime.slot_start, 2.)
        self.assertEqual(env._runtime.deadline, 4.)
        second = env.end_step()
        self.assertTrue(second.epoch_done)
        np.testing.assert_allclose(result.energy_virtual_queue_j, expected_queue)
        env.reset(epoch_index=0)
        np.testing.assert_array_equal(members.energy_virtual_queue_j, 0.)
        np.testing.assert_allclose(result.energy_virtual_queue_j, expected_queue)

    def test_completed_compute_updates_observation_then_end_only_deducts_one_budget(self):
        env = self.make_environment(duration=2., budget=1000.)
        members = env.members
        count = members.member_uav_count
        owner = EntityRef(EntityKind.MEMBER_UAV, 0)
        executor = EntityRef(EntityKind.MEMBER_UAV, 1)
        ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        cycles = members.config.member_core_frequency * .25
        dag = request(owner, ground, inputs=(0., 0.), cycles=(cycles, 0.))
        with patch.object(env, "_generate_dag_requests", return_value=(dag,)):
            env.begin_step(np.zeros((count, 2)))
        after_flight = members.energy_virtual_queue_j.copy()
        ready = env.submit_schedule({dag.task_keys[0]: PlacementDecision(executor)})
        self.assertEqual(ready, (0,))
        runtime = env._runtime
        self.assertEqual(runtime.now, .25)
        p = members.config.member_propulsion
        hover_power = p.u1 + p.u3 * p.u4**.25
        expected = after_flight + .25 * hover_power
        compute_energy = members.config.member_capacitance_factor * cycles * members.config.member_core_frequency**2
        expected[1] += compute_energy
        np.testing.assert_allclose(members.energy_virtual_queue_j, expected)
        observation = env.observe_member(dag.task_keys[1])
        self.assertAlmostEqual(observation[4 + 4 * count + 1], expected[1] / 1000.)
        self.assertEqual(env.submit_schedule({dag.task_keys[1]: PlacementDecision(ground)}), ())
        result = env.end_step()
        self.assertEqual(result.slot_result.finished_dags, (dag.key,))
        expected_compute = np.zeros(count)
        expected_compute[1] = compute_energy
        np.testing.assert_allclose(result.compute_energy_j, expected_compute)
        np.testing.assert_allclose(result.hover_energy_j, 2. * hover_power)
        np.testing.assert_array_equal(result.energy_virtual_queue_j, 0.)
        np.testing.assert_allclose(result.total_energy_j, after_flight + 2. * hover_power + expected_compute)

    def test_auto_closed_runtime_partial_energy_is_not_added_twice(self):
        env = self.make_environment(duration=1.)
        count = env.members.member_uav_count
        owner = EntityRef(EntityKind.MEMBER_UAV, 0)
        executor = EntityRef(EntityKind.MEMBER_UAV, 1)
        ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        frequency = env.members.config.member_core_frequency
        dag = request(owner, ground, cycles=(2. * frequency,))
        with patch.object(env, "_generate_dag_requests", return_value=(dag,)):
            env.begin_step(np.zeros((count, 2)))
        self.assertEqual(env.submit_schedule({dag.task_keys[0]: PlacementDecision(executor)}), ())
        self.assertTrue(env._runtime.closed)
        before = env.members.energy_virtual_queue_j.copy()
        result = env.end_step()
        self.assertAlmostEqual(result.compute_energy_j[1], env.members.config.member_capacitance_factor * frequency**3)
        np.testing.assert_allclose(result.energy_virtual_queue_j, np.maximum(0., before - 145.))


if __name__ == '__main__':
    unittest.main()
