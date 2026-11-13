"""真实环境时隙开始、完整结算和旧运行时隔离。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from env.contracts import DAGRequest, PlacementDecision
from env.env import MRDTSEnv
from tests import test_mrdts_env
from env.types import EntityKind, EntityRef
from env.workload.dag_generator import DAG


class TestSlotLifecycle(unittest.TestCase):
    def setUp(self):
        self.environment = MRDTSEnv(test_mrdts_env.TestMRDTSEnv.settings())
        self.environment.generate_scene()
        self.environment.reset(epoch_index=0)
        self.region, self.users, self.members = self.environment.region, self.environment.users, self.environment.members
        # 本组测试手工构造负载，以验证超时完成和资源隔离。
        workload = patch.object(self.environment, "_generate_dag_requests", return_value=())
        workload.start()
        self.addCleanup(workload.stop)
        self.actions = np.zeros((self.members.member_uav_count, 2))
        self.ground = EntityRef(EntityKind.GROUND_DEVICE, 0)

    def _member_ids(self, region_id):
        return np.flatnonzero(self.members.region_ids[:self.members.member_uav_count] == region_id)

    def _begin(self):
        self.environment.begin_step(self.actions)
        return self.environment._runtime

    def _submit(self, runtime, dag_id=0, cycles=1e7, input_kbit=1.0, execution=None):
        owner = EntityRef(EntityKind.MEMBER_UAV, int(self.environment.user_member_ids[self.ground.index]))
        request = DAGRequest(dag_id, DAG(1, [], np.array([[input_kbit, cycles]]), {}),
                             owner, self.ground)
        key = runtime.register_dags([request])[0]
        runtime.submit_placements({key: PlacementDecision(execution or self.ground)})
        return key

    def _prepare_region_crossing(self):
        grid = self.region.region_map
        row, column = np.argwhere(grid[:, :-1] != grid[:, 1:])[0]
        source, target = int(grid[row, column]), int(grid[row, column + 1])
        member_id = int(self._member_ids(source)[0])
        self.assertGreater(self.members.region_member_counts[source - 1], 1)
        cell_size = self.region.config.cell_size
        self.members.positions[member_id, :2] = (
            (column + 1) * cell_size - 1.0, (row + 0.5) * cell_size
        )
        self.actions[member_id, 0] = 2.0 / (
            self.members.config.max_horizontal_speed * self.members.env_config.flight_duration_s
        )
        return member_id, source, target

    def test_migration_updates_region_counts_and_return_flight_restores_them(self):
        member_id, source, target = self._prepare_region_crossing()
        initial_counts = self.members.region_member_counts.copy()
        expected = initial_counts.copy()
        expected[source - 1] -= 1
        expected[target - 1] += 1

        self._begin()
        self.assertEqual(self.members.region_ids[member_id], target)
        np.testing.assert_array_equal(self.members.region_member_counts, expected)
        self.assertEqual(self.members.region_member_counts.sum(), self.members.member_uav_count)
        self.environment.end_step().slot_result

        self.actions *= -1
        self._begin()
        self.assertEqual(self.members.region_ids[member_id], source)
        np.testing.assert_array_equal(self.members.region_member_counts, initial_counts)
        self.environment.end_step().slot_result

    def test_refresh_keeps_zero_count_for_empty_last_region(self):
        last_region = self.region.config.region_count
        departing = self._member_ids(last_region)
        destination = self._member_ids(1)[0]
        self.members.positions[departing] = self.members.positions[destination]
        expected = self.members.region_member_counts.copy()
        expected[0] += expected[-1]
        expected[-1] = 0
        keep = self.users.region_ids != last_region
        self.users.positions = self.users.positions[keep]
        self.users.region_ids = self.users.region_ids[keep]
        self.users.region_user_counts[-1] = 0

        self.environment._refresh_topology()

        np.testing.assert_array_equal(self.members.region_member_counts, expected)
        self.assertEqual(self.members.region_member_counts.sum(), self.members.member_uav_count)

    def test_begin_refreshes_topology_before_creating_runtime(self):
        original_create = self.environment._create_scheduling_runtime

        def check_topology():
            positions = self.users.positions
            associations = self.environment.user_member_ids
            for user_id, member_id in enumerate(associations):
                self.assertEqual(self.members.region_ids[member_id], self.users.region_ids[user_id])
                candidates = self._member_ids(self.users.region_ids[user_id])
                distances = np.linalg.norm(self.members.positions[candidates, :2] - positions[user_id, :2], axis=1)
                self.assertEqual(member_id, candidates[np.argmin(distances)])
            return original_create()

        # 令区域编号过期，只有先刷新物理拓扑，工厂入口处的检查才会通过。
        self.members.region_ids[:self.members.bs_index] = 1
        with patch.object(self.environment, "_create_scheduling_runtime", side_effect=check_topology):
            violations = self.environment.begin_step(self.actions).flight_violations
        self.assertEqual(violations.dtype, np.bool_)
        self.assertEqual(violations.shape, (self.members.member_uav_count,))
        self.assertEqual(self.environment._runtime.slot_start, 0.0)
        self.assertEqual(self.environment._runtime.deadline, self.environment.env_config.scheduling_duration_s)

    def test_overtime_dag_completes_and_next_slot_starts_after_actual_end(self):
        from methods.scheduling import run_online_schedule

        runtime = self._begin()
        duration = self.environment.env_config.scheduling_duration_s
        ground_frequency = self.users.config.core_frequency
        tail_duration = 0.01 * duration
        tail_cycles = tail_duration * ground_frequency
        expected_end = runtime.slot_start + 2 * duration + 2 * tail_duration
        owner = EntityRef(EntityKind.MEMBER_UAV, int(self.environment.user_member_ids[self.ground.index]))
        request = DAGRequest(0, DAG(3, [(0, 1), (1, 2)],
                             np.array([[1., 2 * duration * ground_frequency],
                                       [1., tail_cycles], [1., tail_cycles]]),
                             {(0, 1): 1., (1, 2): 1.}), owner, self.ground)
        keys = runtime.register_dags([request])
        run_online_schedule(runtime, {owner: keys}, lambda key: PlacementDecision(self.ground))
        result = self.environment.end_step().slot_result
        self.assertIsNone(self.environment._runtime)
        self.assertTrue(runtime.closed)
        self.assertEqual(result.failed_dags, (request.key,))
        self.assertFalse(result.finished_dags)
        self.assertAlmostEqual(result.dags[0].completion_time, expected_end)
        self.assertAlmostEqual(result.slot_end, expected_end)
        self.assertTrue(all(runtime.trace(key).status.value == "finished" for key in keys))
        self.assertFalse(runtime.channels)
        self.assertTrue(all(not server.queue and not server.running for server in runtime.servers.values()))
        new_runtime = self._begin()
        self.assertIsNot(new_runtime, runtime)
        self.assertEqual(new_runtime.slot_start, result.slot_end)
        self.assertFalse(new_runtime.tasks)
        key = self._submit(new_runtime, cycles=tail_cycles)
        next_result = self.environment.end_step().slot_result
        self.assertEqual(new_runtime.trace(key).compute_start_at, result.slot_end)
        self.assertAlmostEqual(new_runtime.trace(key).compute_finish_at, result.slot_end + tail_duration)
        self.assertEqual(len(next_result.finished_dags), 1)

    def test_exact_boundary_completion_allows_next_decision(self):
        runtime = self._begin()
        duration = self.environment.env_config.scheduling_duration_s
        ground_frequency = self.users.config.core_frequency
        tail_duration = 0.01 * duration
        finished = self._submit(runtime, dag_id=0, cycles=duration * ground_frequency)
        owner = EntityRef(EntityKind.MEMBER_UAV, int(self.environment.user_member_ids[self.ground.index]))
        waiting = runtime.register_dags([DAGRequest(
            1, DAG(1, [], np.array([[0., tail_duration * ground_frequency]]), {}), owner, self.ground)])[0]
        self.assertEqual(runtime.advance_to_decision(), (owner.index,))
        self.assertEqual(runtime.now, runtime.deadline)
        runtime.submit_placements({waiting: PlacementDecision(self.ground)})
        result = self.environment.end_step().slot_result
        self.assertEqual(runtime.trace(finished).compute_finish_at, runtime.deadline)
        self.assertEqual(runtime.trace(waiting).compute_start_at, runtime.deadline)
        self.assertAlmostEqual(runtime.trace(waiting).compute_finish_at, runtime.deadline + tail_duration)
        self.assertEqual(result.finished_dags, ((finished.owner_member_id, finished.user_id, finished.dag_id),))
        self.assertEqual(result.failed_dags, ((waiting.owner_member_id, waiting.user_id, waiting.dag_id),))

    def test_advancing_to_nominal_boundary_does_not_close_runtime(self):
        runtime = self._begin()
        duration = self.environment.env_config.scheduling_duration_s
        key = self._submit(runtime, cycles=2 * duration * self.users.config.core_frequency)
        runtime.advance_until(runtime.deadline)
        self.assertFalse(runtime.closed)
        self.assertEqual(runtime.trace(key).status.value, "running")
        result = runtime.finish_slot()
        self.assertAlmostEqual(runtime.trace(key).compute_finish_at, runtime.slot_start + 2 * duration)
        self.assertIs(self.environment.end_step().slot_result, result)
        self.assertIsNone(self.environment._runtime)

    def test_computation_energy_covers_full_overtime_execution(self):
        runtime = self._begin()
        duration = self.environment.env_config.scheduling_duration_s
        member = EntityRef(EntityKind.MEMBER_UAV, int(self.environment.user_member_ids[self.ground.index]))
        key = self._submit(runtime, cycles=2 * duration * self.members.config.member_core_frequency,
                           execution=member)
        runtime.advance_until(runtime.slot_start + 0.1 * duration)
        task = runtime.trace(key)
        self.assertEqual(task.status.value, "running")
        frequency = runtime.servers[member].core_frequencies[task.execution_core_id]
        expected = runtime.servers[member].capacitance_factor * frequency**2 * task.cpu_cycles
        result = self.environment.end_step().slot_result
        self.assertAlmostEqual(task.compute_energy_j, expected)
        self.assertAlmostEqual(result.compute_energy_j, expected)
        self.assertGreater(task.compute_finish_at, runtime.deadline)
        self.assertEqual(len(result.failed_dags), 1)
        self.assertFalse(result.finished_dags)

    def test_position_snapshot_is_independent_of_input_array(self):
        runtime = self._begin()
        position = self.users.positions[0]
        snapshot = position.copy()
        position[0] += 100
        np.testing.assert_array_equal(runtime.entity_positions[self.ground], snapshot)

    def test_duration_configuration_controls_consecutive_windows(self):
        self.environment.env_config = replace(self.environment.env_config, scheduling_duration_s=0.25)
        self.assertEqual(self._begin().deadline, 0.25)
        first = self.environment.end_step().slot_result
        self.assertEqual(first.slot_start, 0.0)
        self.assertEqual(self._begin().deadline, 0.5)
        self.environment.end_step().slot_result


if __name__ == "__main__":
    unittest.main()
