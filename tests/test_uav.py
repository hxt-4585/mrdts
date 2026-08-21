"""UAV 集合基础状态测试。"""

import unittest

import numpy as np

from env.region import Region
from env.uav import MasterUAV, MemberUAV, UAV
from env.user import User


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
    def test_generates_configured_members_at_user_positions_in_their_regions(self):
        """Member 应按用户分布初始化，并保持正确的区域归属和高度。"""
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
            user_positions = users.positions[users.region_ids == region_id, :2]
            self.assertTrue(np.any(np.all(user_positions == position[:2], axis=1)))

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

    def test_selects_the_closest_users_to_each_master_for_initial_positions(self):
        """每个区域的初始 Member 应选取离本区域 Master 最近的用户位置。"""
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()

        members.generate_from_region_and_user(region, users, masters)

        for region_id, count in enumerate(members.region_member_counts, start=1):
            user_positions = users.positions[users.region_ids == region_id, :2]
            master_position = masters.positions[masters.region_ids == region_id, :2][0]
            expected = np.sort(((user_positions - master_position) ** 2).sum(axis=1))[:count]
            actual_positions = members.positions[members.region_ids == region_id, :2]
            actual = np.sort(((actual_positions - master_position) ** 2).sum(axis=1))
            np.testing.assert_allclose(actual, expected)

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

        positions = members.apply_flight_actions(normalized_actions)

        self.assertEqual(members.config.member_slot_energy, 145.0)
        self.assertEqual(members.config.flight_duration, 0.5)
        self.assertEqual(members.config.max_horizontal_speed, 10.0)
        np.testing.assert_allclose(
            positions[: members.bs_index, :2],
            original_positions[: members.bs_index, :2] + np.array([2.0, -1.0]),
        )
        np.testing.assert_array_equal(positions[:, 2], original_positions[:, 2])

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
            np.zeros((members.member_uav_count, 2), dtype=np.float32)
        )

        np.testing.assert_array_equal(members.positions[members.bs_index], original_bs_position)
