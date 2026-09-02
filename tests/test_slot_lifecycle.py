"""真实环境时隙开始、截止结算和旧运行时隔离。"""

import unittest
from unittest.mock import patch

import numpy as np

from env.channel_model import ChannelModel
from env.channel_queue import EntityKind, EntityRef
from env.dag_generator import DAG
from env.environment import Environment
from env.region import Region
from env.uav import MasterUAV, MemberUAV
from env.user import User
from methods.contracts import PlacementDecision


class TestSlotLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.region = Region()
        cls.region.generate()

    def setUp(self):
        self.users = User()
        self.users.generate_from_region(self.region)
        masters = MasterUAV()
        masters.generate_from_region(self.region)
        self.members = MemberUAV()
        self.members.generate_from_region_and_user(self.region, self.users, masters)
        self.environment = Environment(self.members, ChannelModel(), 0.1)
        self.actions = np.zeros((self.members.member_uav_count, 2))
        self.ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        self.assertTrue(callable(getattr(self.environment, "begin_slot", None)),
                        "Environment 必须提供正式 begin_slot 生命周期接口")

    def _begin(self):
        self.environment.begin_slot(self.region, self.users.positions, self.actions)
        return self.environment.runtime

    def _submit(self, runtime, dag_id=0, cycles=1e7, input_kb=1.0, execution=None):
        owner = runtime.ground_owner_members[self.ground]
        return runtime.submit_dag(
            dag_id, DAG(1, [], np.array([[input_kb, cycles]]), {}),
            owner, self.ground, {0: PlacementDecision(execution or self.ground, 0)},
            epoch_start=runtime.now,
        )[0]

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
        self.assertEqual(new_runtime.trace(key).ers_seq, 0)
        next_result = self.environment.end_slot()
        self.assertEqual(new_runtime.trace(key).compute_start_at, 1.0)
        self.assertAlmostEqual(new_runtime.trace(key).compute_finish_at, 1.01)
        self.assertEqual(len(next_result.finished_dags), 1)
        self.assertEqual(result.failed_dags, ((owner.index, 0, 0),))

    def test_closed_reference_cannot_advance_submit_or_refresh(self):
        runtime = self._begin()
        self.environment.end_slot()
        operations = (
            lambda: runtime.advance_until(2.0),
            lambda: self._submit(runtime),
            lambda: runtime.update_entity_positions({self.ground: self.users.positions[0]}),
            lambda: runtime.update_topology(runtime.member_regions, runtime.ground_owner_members),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "关闭"):
                    operation()

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

    def test_invalid_lifecycle_order_does_not_move_members(self):
        with self.assertRaisesRegex(RuntimeError, "时隙"):
            self.environment.end_slot()
        runtime = self._begin()
        positions = self.members.positions.copy()
        with self.assertRaisesRegex(RuntimeError, "时隙"):
            self.environment.begin_slot(self.region, self.users.positions, self.actions + 0.5)
        with self.assertRaisesRegex(RuntimeError, "时隙"):
            self.environment.refresh_slot_topology(self.region, self.users.positions)
        np.testing.assert_array_equal(positions, self.members.positions)
        self.assertIs(self.environment.runtime, runtime)
        self.environment.end_slot()
        with self.assertRaisesRegex(RuntimeError, "时隙"):
            self.environment.end_slot()

    def test_begin_failure_restores_physical_state(self):
        positions = self.members.positions.copy()
        regions = self.members.region_ids.copy()
        counts = self.members.region_member_counts.copy()
        invalid_users = self.users.positions.copy()
        invalid_users[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.environment.begin_slot(self.region, invalid_users, self.actions + 0.5)
        self.assertIsNone(self.environment.runtime)
        np.testing.assert_array_equal(positions, self.members.positions)
        np.testing.assert_array_equal(regions, self.members.region_ids)
        np.testing.assert_array_equal(counts, self.members.region_member_counts)
        self.assertEqual(self._begin().slot_start, 0.0)

    def test_window_rejects_invalid_times_and_future_submission_without_advancement(self):
        runtime = self._begin()
        for at in (-1.0, 1.1, float("nan"), float("inf")):
            with self.subTest(at=at), self.assertRaises(ValueError):
                runtime.advance_until(at)
        owner = runtime.ground_owner_members[self.ground]
        for at in (-1.0, 0.5, 1.0, float("nan")):
            with self.subTest(at=at), self.assertRaises(ValueError):
                runtime.submit_dag(0, DAG(1, [], np.array([[1.0, 1e7]]), {}), owner, self.ground,
                                   {0: PlacementDecision(self.ground, 0)}, epoch_start=at)
        self.assertEqual(runtime.now, 0.0)
        self.assertFalse(runtime.tasks)

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

    def test_topology_and_position_snapshots_cannot_change_after_submission(self):
        runtime = self._begin()
        position = self.users.positions[0].astype(float)
        runtime.update_entity_positions({self.ground: position})
        snapshot = position.copy()
        position[0] += 100
        np.testing.assert_array_equal(runtime.entity_positions[self.ground], snapshot)
        self._submit(runtime)
        with self.assertRaisesRegex(RuntimeError, "拓扑"):
            runtime.update_entity_positions({self.ground: position})
        with self.assertRaisesRegex(RuntimeError, "拓扑"):
            runtime.update_topology(runtime.member_regions, runtime.ground_owner_members)

    def test_duration_configuration_controls_consecutive_windows(self):
        from env.settings import SchedulingConfig

        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SchedulingConfig(value)
        self.assertEqual(SchedulingConfig.default().max_duration_s, 1.0)
        self.environment = Environment(self.members, ChannelModel(), 0.1,
                                       scheduling_config=SchedulingConfig(0.25))
        self.assertEqual(self._begin().deadline, 0.25)
        first = self.environment.end_slot()
        self.assertEqual(first.slot_start, 0.0)
        self.assertEqual(self._begin().deadline, 0.5)
        self.environment.end_slot()


if __name__ == "__main__":
    unittest.main()
