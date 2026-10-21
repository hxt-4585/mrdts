"""真实环境时隙开始、截止结算和旧运行时隔离。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from scripts.randomness import RandomStreams
from env.communication.channel_model import ChannelModel
from env.contracts import PlacementDecision
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User
from env.simulator import Simulator
from env.types import EntityKind, EntityRef
from env.workload.dag_generator import DAG


class TestSlotLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.region = Region(rng=RandomStreams().region)
        cls.region.generate()

    def setUp(self):
        self.users = User(rng=RandomStreams().user)
        self.users.generate_from_region(self.region)
        masters = MasterUAV()
        masters.generate_from_region(self.region)
        self.members = MemberUAV()
        self.members.generate_from_region_and_user(self.region, self.users, masters)
        self.environment = Simulator(self.members, ChannelModel(), 0.1)
        self.actions = np.zeros((self.members.member_uav_count, 2))
        self.ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        self.assertTrue(callable(getattr(self.environment, "begin_slot", None)),
                        "Simulator 必须提供正式 begin_slot 生命周期接口")

    def _begin(self):
        self.environment.begin_slot(self.region, self.users.positions, self.actions)
        return self.environment.runtime

    def _submit(self, runtime, dag_id=0, cycles=1e7, input_kbit=1.0, execution=None):
        owner = runtime.ground_owner_members[self.ground]
        return runtime.submit_dag(
            dag_id, DAG(1, [], np.array([[input_kbit, cycles]]), {}),
            owner, self.ground, {0: PlacementDecision(execution or self.ground, 0)},
            epoch_start=runtime.now,
        )[0]

    def _prepare_region_crossing(self):
        grid = self.region.region_map
        row, column = np.argwhere(grid[:, :-1] != grid[:, 1:])[0]
        source, target = int(grid[row, column]), int(grid[row, column + 1])
        member_id = int(self.environment.member_ids_in_region(source)[0])
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
        self.environment.end_slot()

        self.actions *= -1
        self._begin()
        self.assertEqual(self.members.region_ids[member_id], source)
        np.testing.assert_array_equal(self.members.region_member_counts, initial_counts)
        self.environment.end_slot()

    def test_refresh_keeps_zero_count_for_empty_last_region(self):
        last_region = self.region.config.region_count
        departing = self.environment.member_ids_in_region(last_region)
        destination = self.environment.member_ids_in_region(1)[0]
        self.members.positions[departing] = self.members.positions[destination]
        expected = self.members.region_member_counts.copy()
        expected[0] += expected[-1]
        expected[-1] = 0
        users = self.users.positions[self.users.region_ids != last_region]

        self.environment.refresh_slot_topology(self.region, users)

        np.testing.assert_array_equal(self.members.region_member_counts, expected)
        self.assertEqual(self.members.region_member_counts.sum(), self.members.member_uav_count)

    def test_begin_refreshes_topology_before_creating_runtime(self):
        original_create = self.environment.create_scheduling_runtime

        def check_topology(positions, associations, **kwargs):
            for user_id, member_id in enumerate(associations):
                self.assertEqual(self.members.region_ids[member_id], self.users.region_ids[user_id])
                candidates = self.environment.member_ids_in_region(self.users.region_ids[user_id])
                distances = np.linalg.norm(self.members.positions[candidates, :2] - positions[user_id, :2], axis=1)
                self.assertEqual(member_id, candidates[np.argmin(distances)])
            return original_create(positions, associations, **kwargs)

        # 令区域编号过期，只有先刷新物理拓扑，工厂入口处的检查才会通过。
        self.members.region_ids[:self.members.bs_index] = 1
        with patch.object(self.environment, "create_scheduling_runtime", side_effect=check_topology):
            violations = self.environment.begin_slot(self.region, self.users.positions, self.actions)
        self.assertEqual(violations.dtype, np.bool_)
        self.assertEqual(violations.shape, (self.members.member_uav_count,))
        self.assertEqual(self.environment.runtime.slot_start, 0.0)
        self.assertEqual(self.environment.runtime.deadline, 1.0)

    def test_end_marks_unfinished_work_failed_and_next_slot_is_empty(self):
        runtime = self._begin()
        owner = runtime.ground_owner_members[self.ground]
        keys = runtime.submit_dag(
            0, DAG(3, [], np.array([[1.0, 2e9], [1.0, 1e7], [1e6, 1e7]]), {}),
            owner, self.ground,
            {0: PlacementDecision(self.ground, 0), 1: PlacementDecision(self.ground, 1),
             2: PlacementDecision(runtime.global_bs, 2)}, epoch_start=runtime.slot_start,
        )
        runtime.advance_until(0.5)
        self.assertTrue(runtime.servers[self.ground].running)
        self.assertTrue(runtime.servers[self.ground].queue)
        self.assertTrue(any(channel.active for channel in runtime.channels.values()))
        result = self.environment.end_slot()
        self.assertIsNone(self.environment.runtime)
        self.assertTrue(runtime.closed)
        self.assertEqual(result.failed_dags, ((owner.index, 0, 0),))
        self.assertEqual(result.finished_dags, ())
        self.assertEqual(runtime.dag_runtimes[(owner.index, 0, 0)].failed_at, 1.0)
        self.assertTrue(all(runtime.trace(key).status.value == "failed" for key in keys))
        self.assertFalse(runtime.channels)
        self.assertTrue(all(not server.queue and not server.running for server in runtime.servers.values()))
        self.assertAlmostEqual(result.tx_energy_j, 0.1)

        positions = self.members.positions.copy()
        new_runtime = self._begin()
        self.assertIsNot(new_runtime, runtime)
        np.testing.assert_array_equal(self.members.positions, positions)
        self.assertEqual(new_runtime.slot_start, 1.0)
        self.assertEqual(new_runtime.now, 1.0)
        self.assertFalse(new_runtime.tasks)
        self.assertFalse(new_runtime.dag_runtimes)
        key = self._submit(new_runtime)
        self.assertEqual(new_runtime.trace(key).priority_seq, 0)
        next_result = self.environment.end_slot()
        self.assertEqual(new_runtime.trace(key).compute_start_at, 1.0)
        self.assertAlmostEqual(new_runtime.trace(key).compute_finish_at, 1.01)
        self.assertEqual(len(next_result.finished_dags), 1)
        self.assertEqual(result.failed_dags, ((owner.index, 0, 0),))

    def test_exact_deadline_completion_succeeds_without_starting_waiting_work(self):
        runtime = self._begin()
        finished = self._submit(runtime, dag_id=0, cycles=1e9)
        waiting = self._submit(runtime, dag_id=1)
        result = self.environment.end_slot()
        self.assertEqual(runtime.trace(finished).compute_finish_at, 1.0)
        self.assertIsNone(runtime.trace(waiting).compute_start_at)
        self.assertEqual(len(result.finished_dags), 1)
        self.assertEqual(len(result.failed_dags), 1)

    def test_advancing_to_deadline_closes_runtime_and_end_collects_same_result(self):
        runtime = self._begin()
        runtime.advance_until(runtime.deadline)
        self.assertTrue(runtime.closed)
        result = runtime.finish_slot()
        self.assertIs(self.environment.end_slot(), result)
        self.assertIsNone(self.environment.runtime)

    def test_partial_computation_energy_stops_at_deadline(self):
        runtime = self._begin()
        member = runtime.ground_owner_members[self.ground]
        key = self._submit(runtime, cycles=2e10, execution=member)
        runtime.advance_until(0.1)
        task = runtime.trace(key)
        self.assertEqual(task.status.value, "running")
        frequency = runtime.servers[member].core_frequencies[task.execution_core_id]
        expected = runtime.servers[member].capacitance_factor * frequency**3 * (1.0 - task.compute_start_at)
        result = self.environment.end_slot()
        self.assertAlmostEqual(task.compute_energy_j, expected)
        self.assertAlmostEqual(result.compute_energy_j, expected)
        self.assertIsNone(task.compute_finish_at)

    def test_position_snapshot_is_independent_of_input_array(self):
        runtime = self._begin()
        position = self.users.positions[0].astype(float)
        runtime.update_entity_positions({self.ground: position})
        snapshot = position.copy()
        position[0] += 100
        np.testing.assert_array_equal(runtime.entity_positions[self.ground], snapshot)

    def test_duration_configuration_controls_consecutive_windows(self):
        from env.settings import EnvConfig

        self.assertEqual(EnvConfig.default().scheduling_duration_s, 1.0)
        self.environment = Simulator(self.members, ChannelModel(), 0.1,
                                       env_config=replace(EnvConfig.default(), scheduling_duration_s=0.25))
        self.assertEqual(self._begin().deadline, 0.25)
        first = self.environment.end_slot()
        self.assertEqual(first.slot_start, 0.0)
        self.assertEqual(self._begin().deadline, 0.5)
        self.environment.end_slot()


if __name__ == "__main__":
    unittest.main()
