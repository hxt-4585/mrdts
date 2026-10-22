"""MRDTSEnv 场景生成测试。"""

import unittest
from dataclasses import replace

import numpy as np

from env.contracts import DAGRequest, PlacementDecision
from env.env import MRDTSEnv
from env.runtime.task_runtime import TaskStatus
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.types import EntityKind, EntityRef, TaskKey
from env.workload.dag_generator import DAG


class TestMRDTSEnv(unittest.TestCase):
    @staticmethod
    def settings():
        return {
            "env": EnvConfig.default(),
            "dag": DAGConfig.default(),
            "region": RegionConfig.default(),
            "user": UserConfig.default(),
            "uav": UAVConfig.default(),
            "channel": ChannelConfig.default(),
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
        self.assertEqual(observation.shape, (3 * members.member_uav_count,))
        np.testing.assert_allclose(observation, expected.ravel())

        masters.neighbor_region_ids[2] = (3,)
        other_master_observation = environment.observe_master(1)
        np.testing.assert_array_equal(other_master_observation[:3], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(other_master_observation[3:6], expected[1])

        members.region_ids[2] = 2
        members.positions[2, :2] = [side_length * 0.5, side_length * 0.75]
        refreshed = environment.observe_master(0)
        np.testing.assert_allclose(refreshed[6:9], [0.5, 0.75, 1.0])
        np.testing.assert_array_equal(observation[6:9], [0.0, 0.0, 0.0])

    def test_master_observer_follows_scene_lifecycle_and_returns_independent_arrays(self):
        environment = MRDTSEnv(self.settings())
        environment.generate_scene()
        observer = environment._master_observation

        first = environment.observe_master(0)
        second = environment.observe_master(0)
        self.assertIs(environment._master_observation, observer)
        self.assertIsNot(first, second)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.ndim, 1)
        second[0] = -1.0
        self.assertNotEqual(first[0], second[0])

        environment.reset(epoch_index=1)
        self.assertIs(environment._master_observation, observer)

        environment.generate_scene()
        self.assertIsNot(environment._master_observation, observer)
        self.assertIs(environment._master_observation.members, environment.members)

    @staticmethod
    def register_member_tasks(environment, *, slot_start=0.0):
        environment.user_member_ids = np.full(environment.users.num_users, 2)
        environment._next_slot_start = slot_start
        environment._create_scheduling_runtime()
        runtime = environment._runtime
        config = environment.settings["dag"]
        features = np.zeros((6, 2))
        features[4] = [config.input_data_range[1] * 0.25, config.cpu_range[1] * 0.5]
        features[5] = [config.input_data_range[1] * 0.8, config.cpu_range[1] * 0.2]
        edges = [(parent, 4) for parent in range(4)]
        request = DAGRequest(
            0, DAG(6, edges, features, dict.fromkeys(edges, 0.0)),
            EntityRef(EntityKind.MEMBER_UAV, 2), EntityRef(EntityKind.GROUND_DEVICE, 0),
        )
        keys = runtime.register_dags([request])
        targets = (
            EntityRef(EntityKind.MEMBER_UAV, 1),
            EntityRef(EntityKind.MEMBER_UAV, 1),
            EntityRef(EntityKind.MEMBER_UAV, environment.members.member_uav_count - 1),
            EntityRef(EntityKind.BS, 0),
        )
        for key, target in zip(keys, targets):
            runtime.submit_placements({key: PlacementDecision(target)})
            runtime.advance_to_decision()
        return keys

    def test_member_vector_encodes_task_global_predecessors_identity_and_local_state(self):
        environment = MRDTSEnv(self.settings())
        environment.generate_scene()
        members = environment.members
        count = members.member_uav_count
        members.region_ids[:count] = 3
        members.region_ids[:3] = [1, 2, 1]
        side_length = environment.region.config.side_length
        members.positions[0, :2] = [0.25 * side_length, 0.5 * side_length]
        members.positions[2, :2] = [0.75 * side_length, 0.25 * side_length]
        members.energy_virtual_queue_j[:] = 2 * members.config.member_slot_energy
        members.energy_virtual_queue_j[0] = 0.5 * members.config.member_slot_energy
        keys = self.register_member_tasks(environment)

        observation = environment.observe_member(keys[4])

        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (6 * count + 8,))
        predecessors = np.zeros(count + 2)
        predecessors[[1, count - 1, count]] = 1
        identity = np.zeros(count)
        identity[2] = 1
        positions = np.zeros((count, 2))
        positions[0] = [0.25, 0.5]
        positions[2] = [0.75, 0.25]
        queues = np.zeros(count)
        queues[[0, 2]] = [0.5, 2.0]
        expected = np.concatenate(
            ([0.5, 0.25], predecessors, identity, positions.ravel(), queues, np.zeros(count + 2),
             environment.users.positions[0, :2].astype(float) / side_length)
        )
        np.testing.assert_allclose(observation, expected)
        self.assertEqual(environment._runtime.now, 0.0)
        self.assertIsNone(environment._runtime.tasks[keys[4]].execution_node)

        root = environment.observe_member(keys[5])
        np.testing.assert_allclose(root[:2], [0.2, 0.8])
        np.testing.assert_array_equal(root[2:4 + count], 0.0)
        np.testing.assert_array_equal(root[4 + count:], expected[4 + count:])

        members.region_ids[0] = 2
        members.positions[2, :2] = [0.5 * side_length, 0.75 * side_length]
        members.energy_virtual_queue_j[2] = members.config.member_slot_energy
        refreshed = environment.observe_member(keys[4])
        positions[0] = 0.0
        positions[2] = [0.5, 0.75]
        queues[[0, 2]] = [0.0, 1.0]
        np.testing.assert_allclose(refreshed[4 + 2 * count:4 + 4 * count], positions.ravel())
        np.testing.assert_allclose(refreshed[4 + 4 * count:4 + 5 * count], queues)
        np.testing.assert_array_equal(observation, expected)

    def test_member_observation_includes_own_user_and_local_predecessors(self):
        environment = MRDTSEnv(self.settings())
        environment.generate_scene()
        count = environment.members.member_uav_count
        side_length = environment.region.config.side_length
        environment.users.positions[7, :2] = [side_length * .2, side_length * .8]
        environment.user_member_ids = np.full(environment.users.num_users, 2)
        environment._create_scheduling_runtime()
        runtime = environment._runtime
        ground = EntityRef(EntityKind.GROUND_DEVICE, 7)
        edges = [(parent, 4) for parent in range(4)]
        request = DAGRequest(
            0, DAG(5, edges, np.zeros((5, 2)), dict.fromkeys(edges, 0.0)),
            EntityRef(EntityKind.MEMBER_UAV, 2), ground,
        )
        keys = runtime.register_dags([request])
        for key, target in zip(keys, (ground, ground, request.owner_member, runtime.global_bs)):
            runtime.submit_placements({key: PlacementDecision(target)})
            runtime.advance_to_decision()

        observation = environment.observe_member(keys[4])
        expected = np.zeros(count + 2)
        expected[[2, count, count + 1]] = 1.
        np.testing.assert_array_equal(observation[2:count + 4], expected)
        np.testing.assert_allclose(observation[-2:], [.2, .8])
        self.assertEqual(observation[-3], 0.)

        # 独立服务器快照验证本地等待项读取的是任务所属用户，而非用户 0。
        local = runtime.servers[ground]
        local.core_frequencies = (10.,)
        local.enqueue(keys[4], 5., runtime.now)
        local.dispatch(runtime.now)
        local.enqueue(keys[4], 10., runtime.now)
        other = runtime.servers[EntityRef(EntityKind.GROUND_DEVICE, 0)]
        other.core_frequencies = (1.,)
        other.enqueue(TaskKey(0, 0, 1, 0), 100., runtime.now)
        other.dispatch(runtime.now)
        refreshed = environment.observe_member(keys[4])
        duration = runtime.deadline - runtime.slot_start
        self.assertAlmostEqual(refreshed[-3], 1.5 / duration)
        self.assertEqual(observation[-3], 0.)
        self.assertEqual(runtime.now, 0.)

    def test_member_observer_uses_current_runtime_and_returns_independent_arrays(self):
        environment = MRDTSEnv(self.settings())
        environment.generate_scene()
        keys = self.register_member_tasks(environment)
        first = environment.observe_member(keys[4])
        second = environment.observe_member(keys[4])
        self.assertIsInstance(first, np.ndarray)
        self.assertIsNot(first, second)
        second[0] = -1.0
        self.assertEqual(first[0], 0.5)

        environment.reset(epoch_index=1)
        keys = self.register_member_tasks(environment)
        environment._runtime.tasks[keys[4]].cpu_cycles *= 0.5
        after_reset = environment.observe_member(keys[4])
        self.assertEqual(after_reset[0], 0.25)
        np.testing.assert_array_equal(after_reset[4 + 4 * environment.members.member_uav_count:-2], 0.0)

        environment.generate_scene()
        keys = self.register_member_tasks(environment)
        environment.members.energy_virtual_queue_j[:environment.members.member_uav_count] = (
            environment.members.config.member_slot_energy
        )
        after_generate = environment.observe_member(keys[4])
        self.assertEqual(after_generate[0], 0.5)
        self.assertEqual(after_generate[4 + 4 * environment.members.member_uav_count + 2], 1.0)

    def test_member_compute_wait_includes_bs_and_uses_relative_fixed_window_time(self):
        settings = self.settings()
        settings["env"] = replace(settings["env"], scheduling_duration_s=2.)
        environment = MRDTSEnv(settings)
        environment.generate_scene()
        count = environment.members.member_uav_count
        environment.members.region_ids[:count] = 3
        environment.members.region_ids[:3] = [1, 2, 1]
        keys = self.register_member_tasks(environment, slot_start=10.)
        runtime = environment._runtime
        member = runtime.servers[EntityRef(EntityKind.MEMBER_UAV, 0)]
        hidden = runtime.servers[EntityRef(EntityKind.MEMBER_UAV, 1)]
        bs = runtime.servers[runtime.global_bs]
        # 构建同一观测时刻的服务器快照；BS 使用独立的计算频率。
        for server, frequency, running_cycles, queued_cycles in (
            (member, 10., 6., 4.), (hidden, 10., 50., 10.), (bs, 20., 16., 44.),
        ):
            server.core_frequencies = (frequency,)
            server.enqueue(TaskKey(0, 0, 1, 0), running_cycles, 10.)
            server.dispatch(10.)
            server.enqueue(TaskKey(0, 0, 1, 1), queued_cycles, 10.)
        runtime.now = 10.2

        first = environment.observe_member(keys[4])
        expected = np.zeros(count + 2)
        expected[0] = .8 / 2.
        expected[count] = 2.8 / 2.  # 允许超过 1，不截断截止窗口后的负载。
        np.testing.assert_allclose(first[5 * count + 4:-2], expected)
        self.assertEqual(runtime.now, 10.2)

        runtime.now = 10.5
        later = environment.observe_member(keys[4])
        expected[0] = .5 / 2.
        expected[count] = 2.5 / 2.
        np.testing.assert_allclose(later[5 * count + 4:-2], expected)
        self.assertAlmostEqual(first[-4], 1.4)

    def test_begin_step_moves_members_and_registers_tasks_against_updated_topology(self):
        environment = MRDTSEnv(self.settings())
        environment.generate_scene()
        environment.reset(epoch_index=2)
        members = environment.members
        count = members.member_uav_count
        region = environment.region
        row, column = np.argwhere(region.region_map[:, :-1] != region.region_map[:, 1:])[0]
        boundary_x = (column + 1) * region.config.cell_size
        y = (row + .5) * region.config.cell_size
        source_region = region.get_region_id(boundary_x - .5, y)
        target_region = region.get_region_id(boundary_x + .5, y)
        members.positions[0, :2] = [boundary_x - .5, y]
        members.region_ids[0] = source_region
        members.positions[1, 0] = .1
        environment.users.positions[0, :2] = [boundary_x + .5, y]
        environment.users.region_ids[0] = target_region
        previous = members.positions.copy()
        members.energy_virtual_queue_j[:] = 12.
        actions = np.zeros((count, 2))
        actions[0, 0] = 1. / (members.config.max_horizontal_speed * environment.env_config.flight_duration_s)
        actions[1, 0] = -1.

        result = environment.begin_step(actions)

        self.assertIsNotNone(result)
        np.testing.assert_allclose(members.positions[0, :2], [boundary_x + .5, y])
        np.testing.assert_array_equal(members.positions[1], previous[1])
        np.testing.assert_array_equal(members.positions[:, 2], previous[:, 2])
        np.testing.assert_array_equal(members.positions[-1], previous[-1])
        violations = np.zeros(count, dtype=bool)
        violations[1] = True
        np.testing.assert_array_equal(result.flight_violations, violations)
        np.testing.assert_array_equal(environment.flight_violations, violations)
        speed = np.zeros(count)
        speed[0] = 1. / environment.env_config.flight_duration_s
        np.testing.assert_allclose(
            members.energy_virtual_queue_j,
            12. + members.propulsion_power(speed) * environment.env_config.flight_duration_s,
        )
        self.assertEqual(members.region_ids[0], target_region)
        self.assertEqual(environment.user_member_ids[0], 0)
        expected_regions = np.array([region.get_region_id(*position[:2]) for position in members.positions[:count]])
        np.testing.assert_array_equal(members.region_ids[:count], expected_regions)
        np.testing.assert_array_equal(
            members.region_member_counts,
            np.bincount(expected_regions, minlength=region.config.region_count + 1)[1:],
        )

        runtime = environment._runtime
        self.assertEqual(len(result.requests), environment.users.num_users)
        self.assertEqual(result.ready_member_ids, tuple(sorted(set(environment.user_member_ids))))
        self.assertEqual(len(runtime.tasks), sum(request.dag.node_num for request in result.requests))
        for user_id, request in enumerate(result.requests):
            candidates = np.flatnonzero(expected_regions == region.get_region_id(*environment.users.positions[user_id, :2]))
            nearest = candidates[np.argmin(np.linalg.norm(
                members.positions[candidates, :2] - environment.users.positions[user_id, :2], axis=1,
            ))]
            self.assertEqual(request.owner_member.index, nearest)
            self.assertEqual(request.ground_device.index, user_id)
            self.assertEqual(runtime.ground_owner_members[request.ground_device], request.owner_member)
            np.testing.assert_array_equal(runtime.entity_positions[request.owner_member], members.positions[nearest])
            for key in request.task_keys:
                self.assertEqual(runtime.tasks[key].cpu_cycles, request.dag.node_features[key.node_id, 1])
                self.assertEqual(runtime.tasks[key].input_bits, request.dag.node_features[key.node_id, 0] * 1000.)
        self.assertTrue(all(task.status is TaskStatus.UNSCHEDULED for task in runtime.tasks.values()))
        self.assertFalse(runtime.channels)
        self.assertFalse(runtime.active_tasks)
        self.assertTrue(all(not server.queue and not server.running for server in runtime.servers.values()))
        key = next(key for key, task in runtime.tasks.items() if not task.predecessors)
        observation = environment.observe_member(key)
        self.assertEqual(observation.shape, (6 * count + 8,))
        np.testing.assert_array_equal(observation[5 * count + 4:-2], 0.)
        self.assertEqual(runtime.now, 0.)

    def test_begin_step_uses_configured_resources_and_current_window_without_advancing_step(self):
        settings = self.settings()
        settings["user"] = replace(settings["user"], total_users=8, transmit_power=.3, core_frequency=1.7e9)
        settings["env"] = replace(settings["env"], scheduling_duration_s=2.5)
        settings["channel"] = replace(settings["channel"], air_to_air_bandwidth_mhz=17.)
        environment = MRDTSEnv(settings)
        environment.generate_scene()
        environment.reset(epoch_index=0)
        environment.step_index = 3
        environment._next_slot_start = 7.5

        result = environment.begin_step(np.zeros((environment.members.member_uav_count, 2)))

        self.assertIsNotNone(result)
        runtime = environment._runtime
        self.assertEqual(runtime.now, 7.5)
        self.assertEqual(runtime.slot_start, 7.5)
        self.assertEqual(runtime.deadline, 10.)
        self.assertEqual(environment.step_index, 3)
        self.assertEqual(environment._next_slot_start, 7.5)
        self.assertEqual([request.dag_id for request in result.requests], list(range(24, 32)))
        self.assertIs(runtime.channel_model.config, settings["channel"])
        ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        self.assertEqual(runtime.transmit_powers[ground], .3)
        self.assertEqual(runtime.servers[ground].core_frequencies, (1.7e9,))
        for member_id in range(environment.members.member_uav_count):
            server = runtime.servers[EntityRef(EntityKind.MEMBER_UAV, member_id)]
            self.assertEqual(server.core_frequencies, (settings["uav"].member_core_frequency,) * settings["uav"].member_core_count)
        self.assertEqual(runtime.servers[runtime.global_bs].core_frequencies,
                         (settings["uav"].bs_core_frequency,) * settings["uav"].bs_core_count)
        self.assertLess(len(result.ready_member_ids), environment.members.member_uav_count)

    def test_begin_step_recreates_workload_and_replays_it_after_reset(self):
        settings = self.settings()
        settings["user"] = replace(settings["user"], total_users=8)
        environment = MRDTSEnv(settings)
        environment.generate_scene()
        environment.reset(epoch_index=3)
        actions = np.zeros((environment.members.member_uav_count, 2))
        first = environment.begin_step(actions)
        self.assertIsNotNone(first)
        old_runtime = environment._runtime
        environment.end_step()
        second = environment.begin_step(actions)
        self.assertIsNot(environment._runtime, old_runtime)
        self.assertTrue(set(old_runtime.tasks).isdisjoint(environment._runtime.tasks))
        self.assertEqual(environment._runtime.now, old_runtime.deadline)
        self.assertFalse(np.array_equal(first.requests[0].dag.node_features, second.requests[0].dag.node_features))

        environment.reset(epoch_index=3)
        self.assertIsNone(environment.flight_violations)
        replayed = environment.begin_step(actions)
        for original, replay in zip(first.requests, replayed.requests):
            self.assertEqual(original.key, replay.key)
            self.assertEqual(original.dag.edges, replay.dag.edges)
            np.testing.assert_array_equal(original.dag.node_features, replay.dag.node_features)
            self.assertEqual(original.dag.edge_features, replay.dag.edge_features)

        old_members = environment.members
        environment.generate_scene()
        environment.reset(epoch_index=4)
        environment.members.positions[0, 0] += .25
        regenerated = environment.begin_step(actions)
        self.assertIsNot(environment.members, old_members)
        self.assertFalse(np.array_equal(first.requests[0].dag.node_features, regenerated.requests[0].dag.node_features))
        np.testing.assert_array_equal(
            environment._runtime.entity_positions[EntityRef(EntityKind.MEMBER_UAV, 0)],
            environment.members.positions[0],
        )


if __name__ == "__main__":
    unittest.main()
