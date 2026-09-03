"""UAV 集合基础状态测试。"""

import unittest
from dataclasses import replace

import numpy as np

from env.region import Region
from env.uav import MasterUAV, MemberUAV, UAV
from env.user import User
from env.settings import UAVConfig, UserConfig


class TestUAV(unittest.TestCase):
    def test_initializes_vectorized_positions_and_region_ids(self):
        """UAV 集合应保留与 User 一致的批量位置和区域编号。"""
        positions = np.array([[10.0, 20.0, 100.0], [30.0, 40.0, 120.0]], dtype=np.float32)
        region_ids = np.array([1, 3], dtype=np.int32)

        uavs = UAV(positions, region_ids)

        np.testing.assert_array_equal(uavs.positions, positions)
        np.testing.assert_array_equal(uavs.region_ids, region_ids)
        self.assertEqual(uavs.num_uavs, 2)

    def test_rejects_positions_without_three_coordinates(self):
        """每架 UAV 必须有 x、y、z 三个坐标。"""
        positions = np.array([[10.0, 20.0]], dtype=np.float32)
        region_ids = np.array([1], dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "shape"):
            UAV(positions, region_ids)

    def test_rejects_region_ids_with_a_different_uav_count(self):
        """位置数组和区域编号数组必须描述同一批 UAV。"""
        positions = np.array([[10.0, 20.0, 100.0]], dtype=np.float32)
        region_ids = np.array([1, 2], dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "数量"):
            UAV(positions, region_ids)

    def test_updates_selected_uav_positions_in_batch(self):
        """位置更新应支持按 UAV 索引批量写入。"""
        uavs = UAV(
            np.array([[10.0, 20.0, 100.0], [30.0, 40.0, 120.0]], dtype=np.float32),
            np.array([1, 2], dtype=np.int32),
        )

        uavs.update_positions(
            np.array([1], dtype=np.int32),
            np.array([[35.0, 45.0, 120.0]], dtype=np.float32),
        )

        np.testing.assert_array_equal(
            uavs.positions,
            np.array([[10.0, 20.0, 100.0], [35.0, 45.0, 120.0]], dtype=np.float32),
        )

    def test_updates_selected_uav_region_ids_in_batch(self):
        """跨区域迁移后应能批量更新区域编号。"""
        uavs = UAV(
            np.array([[10.0, 20.0, 100.0], [30.0, 40.0, 120.0]], dtype=np.float32),
            np.array([1, 2], dtype=np.int32),
        )

        uavs.update_region_ids(
            np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)
        )

        np.testing.assert_array_equal(uavs.region_ids, np.array([2, 3], dtype=np.int32))


class TestMasterUAV(unittest.TestCase):
    def test_generates_one_master_for_each_region_from_shared_uav_config(self):
        """Master 数量应来自 UAV 配置且与区域数量一一对应。"""
        region = Region()
        region.generate()
        masters = MasterUAV()

        masters.generate_from_region(region)

        self.assertEqual(masters.num_uavs, region.config.region_count)
        np.testing.assert_array_equal(
            masters.region_ids,
            np.arange(1, region.config.region_count + 1, dtype=np.int32),
        )
        self.assertTrue((masters.positions[:, 2] == masters.config.master_altitude).all())
        for position, region_id in zip(masters.positions, masters.region_ids):
            self.assertEqual(region.get_region_id(position[0], position[1]), region_id)


class TestMemberUAV(unittest.TestCase):
    @staticmethod
    def _generate_members(config=None):
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV(config=config)
        members.generate_from_region_and_user(region, users, masters)
        return members

    def test_generates_separated_members_near_masters_in_their_regions(self):
        """Member 应位于本区域 Master 附近，且水平间距至少一格。"""
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()

        members.generate_from_region_and_user(region, users, masters)

        self.assertEqual(members.num_uavs, members.config.member_uav_count + 1)
        self.assertEqual(members.region_member_counts.sum(), members.member_uav_count)
        self.assertTrue(
            (members.positions[: members.bs_index, 2] == members.config.member_altitude).all()
        )
        for position, region_id in zip(
            members.positions[: members.bs_index], members.region_ids[: members.bs_index]
        ):
            self.assertEqual(region.get_region_id(position[0], position[1]), region_id)
            master_position = masters.positions[masters.region_ids == region_id, :2][0]
            self.assertGreaterEqual(np.linalg.norm(position[:2] - master_position), region.config.cell_size)
        for region_id, count in enumerate(members.region_member_counts, start=1):
            xy = members.positions[members.region_ids == region_id, :2]
            master_xy = masters.positions[masters.region_ids == region_id, :2][0]
            # 默认场景的 Master 周围空间充分，全部 Member 都应处在近邻网格内。
            self.assertTrue((np.linalg.norm(xy - master_xy, axis=1) <= 2 * region.config.cell_size).all())
            distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
            self.assertTrue((distances[np.triu_indices(int(count), 1)] >= region.config.cell_size).all())

    def test_appends_bs_at_center_with_zero_region_id_and_configured_cores(self):
        """最后一行应为固定在区域中心的 BS，并拥有独立的计算资源。"""
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()

        members.generate_from_region_and_user(region, users, masters)

        self.assertEqual(members.bs_index, members.config.member_uav_count)
        self.assertEqual(members.member_uav_count, members.config.member_uav_count)
        self.assertEqual(members.num_uavs, members.config.member_uav_count + 1)
        np.testing.assert_allclose(
            members.positions[members.bs_index],
            np.array([500.0, 500.0, 25.0], dtype=np.float32),
        )
        self.assertEqual(members.region_ids[members.bs_index], 0)
        np.testing.assert_array_equal(
            members.core_counts,
            np.array([2] * members.config.member_uav_count + [4], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            members.core_frequencies[: members.bs_index, :2],
            np.full((members.member_uav_count, 2), 10e9),
        )
        np.testing.assert_array_equal(
            members.core_frequencies[members.bs_index], np.full(4, 12e9)
        )

    def test_initial_positions_do_not_depend_on_user_coordinates(self):
        """用户坐标改变但区域人数不变时，Member 初始位置应完全一致。"""
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()

        members.generate_from_region_and_user(region, users, masters)

        original = members.positions.copy()
        for region_id in masters.region_ids:
            cells = np.argwhere(region.region_map == region_id)
            xy = (cells[:, ::-1] + .5) * region.config.cell_size
            master_xy = masters.positions[masters.region_ids == region_id, :2][0]
            farthest = xy[np.argmax(np.sum((xy - master_xy) ** 2, axis=1))]
            users.positions[users.region_ids == region_id, :2] = farthest
        members.generate_from_region_and_user(region, users, masters)
        np.testing.assert_array_equal(members.positions, original)

    def test_one_user_three_members_can_run_ers_after_zero_flight(self):
        """回归：每区域 1 用户、3 Member 不重合，ERS 后可正常本地执行。"""
        from env.channel_model import ChannelModel
        from env.channel_queue import EntityKind, EntityRef
        from env.dag_generator import DAG
        from env.environment import Environment
        from methods.contracts import DAGRequest, PlacementDecision
        from methods.ers import ERS

        region = Region()
        region.generate()
        count = region.config.region_count
        users = User(replace(UserConfig.default(), total_users=count, min_users_per_region=1,
                             area_fluctuation=0.))
        users.generate_from_region(region)
        config = replace(UAVConfig.default(), master_uav_count=count,
                         member_uav_count=3 * count, min_members_per_region=3)
        masters = MasterUAV(config)
        masters.generate_from_region(region)
        members = MemberUAV(config)
        members.generate_from_region_and_user(region, users, masters)
        xy = members.positions[:members.bs_index, :2]
        distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        self.assertGreaterEqual(distances[np.triu_indices(len(xy), 1)].min(), region.config.cell_size)

        environment = Environment(members, ChannelModel(), users.config.transmit_power,
                                  users.config.core_frequency)
        environment.begin_slot(region, users.positions, np.zeros((members.member_uav_count, 2)))
        runtime = environment.runtime
        requests = [DAGRequest(
            0, DAG(2, [(0, 1)], np.array([[1., 1e7], [1., 1e7]]), {(0, 1): 1.}),
            EntityRef(EntityKind.MEMBER_UAV, int(environment.user_member_ids[user_id])),
            EntityRef(EntityKind.GROUND_DEVICE, user_id)) for user_id in range(count)]
        plan = ERS(runtime).plan(requests)
        runtime.submit_dags(requests, {
            key: PlacementDecision(EntityRef(EntityKind.GROUND_DEVICE, key.user_id), seq)
            for seq, key in enumerate(plan.order)
        }, runtime.now)
        result = environment.end_slot()
        self.assertEqual(len(result.finished_dags), count)
        self.assertFalse(result.failed_dags)

    def test_applies_horizontal_velocity_for_configured_flight_duration(self):
        """Member 应将归一化动作缩放为最大速度后更新水平位置。"""
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()
        members.generate_from_region_and_user(region, users, masters)
        original_positions = members.positions.copy()
        normalized_actions = np.tile(
            np.array([0.4, -0.2], dtype=np.float32), (members.member_uav_count, 1)
        )

        violations = members.apply_flight_actions(
            normalized_actions, region.config.side_length
        )

        self.assertEqual(members.config.member_slot_energy, 145.0)
        self.assertEqual(members.config.flight_duration, 0.5)
        self.assertEqual(members.config.max_horizontal_speed, 10.0)
        self.assertFalse(violations.any())
        np.testing.assert_allclose(
            members.positions[: members.bs_index, :2],
            original_positions[: members.bs_index, :2] + np.array([2.0, -1.0]),
        )
        np.testing.assert_array_equal(members.positions[:, 2], original_positions[:, 2])

    def test_flight_actions_leave_bs_position_unchanged(self):
        """Member 飞行仅改变 Member 行，不得移动最后一行的 BS。"""
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()
        members.generate_from_region_and_user(region, users, masters)
        original_bs_position = members.positions[members.bs_index].copy()

        members.apply_flight_actions(
            np.zeros((members.member_uav_count, 2), dtype=np.float32),
            region.config.side_length,
        )

        np.testing.assert_array_equal(members.positions[members.bs_index], original_bs_position)

    def test_rejects_each_out_of_bounds_flight_and_returns_violation_mask(self):
        """越界 Member 保持原位，同时不阻塞同批次内的合法 Member。"""
        members = self._generate_members()
        original_positions = members.positions.copy()
        members.positions[0, :2] = np.array([1.0, 1.0], dtype=np.float32)
        original_positions[0] = members.positions[0]
        actions = np.zeros((members.member_uav_count, 2), dtype=np.float32)
        actions[0] = [-1.0, 0.0]
        actions[1] = [0.2, 0.0]

        violations = members.apply_flight_actions(
            actions, Region().config.side_length
        )

        self.assertEqual(violations.shape, (members.member_uav_count,))
        self.assertEqual(violations.dtype, np.bool_)
        self.assertTrue(violations[0])
        self.assertFalse(violations[1])
        np.testing.assert_array_equal(members.positions[0], original_positions[0])
        np.testing.assert_allclose(
            members.positions[1, :2], original_positions[1, :2] + [1.0, 0.0]
        )
        np.testing.assert_array_equal(
            members.positions[members.bs_index], original_positions[members.bs_index]
        )

    def test_does_not_store_runtime_core_availability(self):
        """Member 只保存静态计算能力，不保存时隙内核心占用状态。"""
        members = self._generate_members()

        self.assertFalse(hasattr(members, "core_available_at"))
        self.assertFalse(hasattr(members, "schedule_computation"))
