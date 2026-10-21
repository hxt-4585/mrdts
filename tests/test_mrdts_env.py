"""MRDTSEnv 场景生成测试。"""

import unittest

import numpy as np

from env.env import MRDTSEnv
from env.settings import DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig


class TestMRDTSEnv(unittest.TestCase):
    @staticmethod
    def settings():
        return {
            "env": EnvConfig.default(),
            "dag": DAGConfig.default(),
            "region": RegionConfig.default(),
            "user": UserConfig.default(),
            "uav": UAVConfig.default(),
        }

    def test_generate_scene_builds_reproducible_entities_and_saves_initial_members(self):
        settings = self.settings()
        first = MRDTSEnv(settings)
        second = MRDTSEnv(settings)

        self.assertIsNone(first.region)
        first.generate_scene()
        second.generate_scene()

        self.assertEqual(first.region.region_map.shape, (settings["region"].grid_size,) * 2)
        self.assertEqual(first.users.num_users, settings["user"].total_users)
        self.assertEqual(first.masters.num_uavs, settings["uav"].master_uav_count)
        self.assertEqual(first.members.member_uav_count, settings["uav"].member_uav_count)
        np.testing.assert_array_equal(first.region.region_map, second.region.region_map)
        np.testing.assert_array_equal(first.users.positions, second.users.positions)
        np.testing.assert_array_equal(first.masters.positions, second.masters.positions)
        np.testing.assert_array_equal(first.members.positions, second.members.positions)

        initial_position = first.members.positions[0].copy()
        first.members.positions[0, 0] += 1
        np.testing.assert_array_equal(first._initial_member_positions[0], initial_position)

    def test_reset_restores_members_and_replays_the_selected_epoch(self):
        environment = MRDTSEnv(self.settings())
        environment.generate_scene()
        region = environment.region
        users = environment.users
        masters = environment.masters
        original_positions = environment.members.positions.copy()
        original_region_ids = environment.members.region_ids.copy()
        original_counts = environment.members.region_member_counts.copy()

        environment.members.positions[0, 0] += 10
        environment.members.region_ids[0] = 0
        environment.members.region_member_counts[0] = 0
        environment.members.energy_virtual_queue_j[:] = 12.0
        environment.reset(epoch_index=3)

        self.assertIs(environment.region, region)
        self.assertIs(environment.users, users)
        self.assertIs(environment.masters, masters)
        np.testing.assert_array_equal(environment.members.positions, original_positions)
        np.testing.assert_array_equal(environment.members.region_ids, original_region_ids)
        np.testing.assert_array_equal(environment.members.region_member_counts, original_counts)
        np.testing.assert_array_equal(environment.members.energy_virtual_queue_j, 0.0)
        self.assertEqual(environment.epoch_index, 3)
        self.assertEqual(environment.step_index, 0)
        self.assertEqual(environment._next_slot_start, 0.0)
        self.assertIsNone(environment._runtime)
        self.assertIsNone(environment.user_member_ids)

        first_dag = environment.dag_generator.generate_single_dag()
        environment.reset(epoch_index=3)
        replayed_dag = environment.dag_generator.generate_single_dag()
        self.assertEqual(first_dag.edges, replayed_dag.edges)
        np.testing.assert_array_equal(first_dag.node_features, replayed_dag.node_features)

        environment.reset(epoch_index=4)
        next_epoch_dag = environment.dag_generator.generate_single_dag()
        self.assertFalse(np.array_equal(first_dag.node_features, next_epoch_dag.node_features))

    def test_observe_master_normalizes_visible_members_and_zeros_hidden_members(self):
        environment = MRDTSEnv(self.settings())
        environment.generate_scene()
        masters = environment.masters
        members = environment.members
        side_length = environment.region.config.side_length
        slot_energy = members.config.member_slot_energy

        masters.neighbor_region_ids[1] = (2,)
        members.region_ids[: members.member_uav_count] = 3
        members.region_ids[:3] = [1, 2, 3]
        members.positions[0, :2] = [side_length * 0.25, side_length * 0.5]
        members.positions[1, :2] = [side_length * 0.75, side_length * 0.25]
        members.energy_virtual_queue_j[:3] = [slot_energy * 0.5, slot_energy * 1.25, slot_energy]

        observation = environment.observe_master(0)

        expected = np.zeros((members.member_uav_count, 3))
        expected[0] = [0.25, 0.5, 0.5]
        expected[1] = [0.75, 0.25, 1.25]
        self.assertIsInstance(observation, np.ndarray)
        np.testing.assert_allclose(observation, expected)

        masters.neighbor_region_ids[2] = (3,)
        other_master_observation = environment.observe_master(1)
        np.testing.assert_array_equal(other_master_observation[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(other_master_observation[1], expected[1])

        members.region_ids[2] = 2
        members.positions[2, :2] = [side_length * 0.5, side_length * 0.75]
        refreshed = environment.observe_master(0)
        np.testing.assert_allclose(refreshed[2], [0.5, 0.75, 1.0])
        np.testing.assert_array_equal(observation[2], [0.0, 0.0, 0.0])

    def test_master_observer_follows_scene_lifecycle_and_returns_independent_arrays(self):
        environment = MRDTSEnv(self.settings())
        environment.generate_scene()
        observer = environment._master_observation

        first = environment.observe_master(0)
        second = environment.observe_master(0)
        self.assertIs(environment._master_observation, observer)
        self.assertIsNot(first, second)
        np.testing.assert_array_equal(first, second)
        second[0, 0] = -1.0
        self.assertNotEqual(first[0, 0], second[0, 0])

        environment.reset(epoch_index=1)
        self.assertIs(environment._master_observation, observer)

        environment.generate_scene()
        self.assertIsNot(environment._master_observation, observer)
        self.assertIs(environment._master_observation.members, environment.members)


if __name__ == "__main__":
    unittest.main()
