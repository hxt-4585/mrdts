"""飞行、事件队列更新和跨窗口结算的环境集成测试。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.step.reward import MasterReward, MemberReward
from env.types import EntityKind, EntityRef
from tests import test_mrdts_env
from tests.test_serial_runtime import request


class TestEndStep(unittest.TestCase):
    def test_member_completion_metrics_keep_pre_budget_queue_snapshot(self):
        env = self.make_environment(duration=2.)
        count = env.members.member_uav_count
        owner = EntityRef(EntityKind.MEMBER_UAV, 0)
        executor = EntityRef(EntityKind.MEMBER_UAV, 1)
        ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        frequency = env.members.config.member_core_frequency
        dag = request(owner, ground, inputs=(0., 0.), cycles=(.25 * frequency,) * 2)
        env.members.energy_virtual_queue_j[:] = 10.
        with patch.object(env, '_generate_dag_requests', return_value=(dag,)):
            env.begin_step(np.zeros((count, 2)))
        runtime = env._runtime
        expected_delay_gain = .25 * frequency / runtime.servers[ground].core_frequencies[0] - .25
        after_flight = env.members.energy_virtual_queue_j.copy()
        hover_power = float(env.members.propulsion_power(0.))
        compute = env.members.config.member_capacitance_factor * .25 * frequency**3
        env.submit_schedule({dag.task_keys[0]: PlacementDecision(executor)})
        metrics = env.pop_member_rewards()[dag.task_keys[0]]
        self.assertIsInstance(metrics, MemberReward)
        expected_queue = after_flight + hover_power * .25
        expected_queue[1] += compute
        np.testing.assert_allclose(metrics.compute_energy_queue(), expected_queue)
        expected_energy = np.zeros(count)
        expected_energy[1] = compute
        np.testing.assert_allclose(metrics.compute_energy(), expected_energy)
        self.assertAlmostEqual(metrics.compute_delay(), expected_delay_gain)
        self.assertEqual(env.pop_member_rewards(), {})
        env.submit_schedule({dag.task_keys[1]: PlacementDecision(owner)})
        end = env.end_step()
        self.assertEqual(set(end.member_rewards), {dag.task_keys[1]})
        final_metrics = end.member_rewards[dag.task_keys[1]]
        self.assertAlmostEqual(final_metrics.compute_delay(), expected_delay_gain)
        expected_final = after_flight + hover_power * .5
        expected_final[[0, 1]] += compute
        np.testing.assert_allclose(final_metrics.compute_energy_queue(), expected_final)
        np.testing.assert_allclose(metrics.compute_energy_queue(), expected_queue)
        self.assertFalse(np.shares_memory(metrics.compute_energy_queue(),
                                         env.members.energy_virtual_queue_j))
        env.reset(epoch_index=0)
        np.testing.assert_allclose(final_metrics.compute_energy_queue(), expected_final)

    def test_simultaneous_completions_see_all_energy_at_that_time(self):
        env = self.make_environment(duration=1.)
        count = env.members.member_uav_count
        owners = [EntityRef(EntityKind.MEMBER_UAV, i) for i in (0, 1)]
        cycles = .25 * env.members.config.member_core_frequency
        dags = tuple(request(owner, EntityRef(EntityKind.GROUND_DEVICE, i), cycles=(cycles,))
                     for i, owner in enumerate(owners))
        with patch.object(env, '_generate_dag_requests', return_value=dags):
            env.begin_step(np.zeros((count, 2)))
        env.submit_schedule({dag.task_keys[0]: PlacementDecision(owner)
                             for dag, owner in zip(dags, owners)})
        expected = env.members.energy_virtual_queue_j.copy()
        results = env.pop_member_rewards()
        self.assertEqual(len(results), 2)
        for metrics in results.values():
            np.testing.assert_allclose(metrics.compute_energy_queue(), expected)

    def test_member_rewards_are_drained_online_or_preserved_at_end_step(self):
        env = self.make_environment(duration=2.)
        owner = EntityRef(EntityKind.MEMBER_UAV, 0)
        ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        cycles = env.members.config.member_core_frequency * .25
        expected_delay_gain = cycles / env.users.config.core_frequency - .25
        dag = request(owner, ground, inputs=(0., 0.), cycles=(cycles, cycles))
        actions = np.zeros((env.members.member_uav_count, 2))
        with patch.object(env, "_generate_dag_requests", return_value=(dag,)):
            env.begin_step(actions)
        self.assertEqual(env.pop_member_rewards(), {})
        env.submit_schedule({dag.task_keys[0]: PlacementDecision(owner)})
        first = env.pop_member_rewards()
        self.assertAlmostEqual(first[dag.task_keys[0]].compute_delay(), expected_delay_gain)
        self.assertEqual(env.pop_member_rewards(), {})
        env.submit_schedule({dag.task_keys[1]: PlacementDecision(owner)})
        result = env.end_step()
        self.assertAlmostEqual(result.member_rewards[dag.task_keys[1]].compute_delay(), expected_delay_gain)
        self.assertEqual(result.master_reward, -.25)
        with patch.object(env, "_generate_dag_requests", return_value=(dag,)):
            env.begin_step(actions)
        self.assertEqual(env.pop_member_rewards(), {})
        self.assertEqual(env.end_step().member_rewards, {})
        self.assertAlmostEqual(first[dag.task_keys[0]].compute_delay(), expected_delay_gain)
        self.assertAlmostEqual(result.member_rewards[dag.task_keys[1]].compute_delay(), expected_delay_gain)

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
        self.assertEqual(len(result.slot_result.failed_dags), 0)
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

    def test_overtime_completion_energy_is_not_added_twice(self):
        env = self.make_environment(duration=1.)
        count = env.members.member_uav_count
        owner = EntityRef(EntityKind.MEMBER_UAV, 0)
        executor = EntityRef(EntityKind.MEMBER_UAV, 1)
        ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        frequency = env.members.config.member_core_frequency
        initial_queue = np.arange(count, dtype=float) * 10.
        env.members.energy_virtual_queue_j[:] = initial_queue
        dag = request(owner, ground, cycles=(2. * frequency,))
        with patch.object(env, "_generate_dag_requests", return_value=(dag,)):
            env.begin_step(np.zeros((count, 2)))
        self.assertEqual(env.submit_schedule({dag.task_keys[0]: PlacementDecision(executor)}), ())
        self.assertFalse(env._runtime.closed)
        before = env.members.energy_virtual_queue_j.copy()
        result = env.end_step()
        self.assertAlmostEqual(result.compute_energy_j[1], 2. * env.members.config.member_capacitance_factor * frequency**3)
        np.testing.assert_allclose(result.energy_virtual_queue_j, np.maximum(0., before - 145.))
        reward = MasterReward(env.env_config.scheduling_duration_s)
        energy = reward.compute_energy(result)
        queues = reward.compute_energy_queue(result)
        np.testing.assert_allclose(
            energy,
            result.flight_energy_j + result.hover_energy_j
            + result.compute_energy_j + result.tx_energy_j,
        )
        np.testing.assert_allclose(queues, np.maximum(0., initial_queue + energy - 145.))
        self.assertEqual(energy.shape, (count,))
        self.assertEqual(queues.shape, (count,))
        self.assertGreater(queues[1], 0.)
        self.assertEqual(reward.compute_delay(result.slot_result), -2.)
        env.reset(epoch_index=0)
        np.testing.assert_allclose(
            reward.compute_energy_queue(result), np.maximum(0., initial_queue + energy - 145.),
        )


if __name__ == '__main__':
    unittest.main()
