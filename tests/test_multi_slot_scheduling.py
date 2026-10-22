"""十个时隙的随机飞行与随机卸载集成测试。"""

import unittest

import numpy as np

from scripts.randomness import RandomStreams
from env.communication.channel_model import ChannelModel
from env.contracts import DAGRequest, PlacementDecision
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User
from env.runtime.task_runtime import TaskStatus
from env.simulator import Simulator
from env.types import EntityKind, EntityRef
from env.workload.dag_generator import DAG
from methods.random.ers import ERS
from methods.scheduling import run_online_schedule


class TestMultiSlotScheduling(unittest.TestCase):
    """随机策略不应破坏独立信道、依赖与时隙 deadline 约束。"""

    SLOT_COUNT = 10
    DAGS_PER_SLOT = 2

    def setUp(self):
        self.region = Region(rng=RandomStreams().region)
        self.region.generate()
        self.users = User(rng=RandomStreams().user)
        self.users.generate_from_region(self.region)
        masters = MasterUAV()
        masters.generate_from_region(self.region)
        self.members = MemberUAV()
        self.members.generate_from_region_and_user(self.region, self.users, masters)
        self.environment = Simulator(
            self.members,
            ChannelModel(),
            user_transmit_power=self.users.config.transmit_power,
            user_core_frequency=self.users.config.core_frequency,
        )
        self.max_duration_s = self.environment.env_config.scheduling_duration_s
        self.rng = RandomStreams().scheduling

    def test_random_flight_and_random_placement_finish_two_dags_per_slot_before_deadline(self):
        for slot_id in range(self.SLOT_COUNT):
            slot_start = slot_id * self.max_duration_s
            self._start_slot(
                self.rng.uniform(-1.0, 1.0, size=(self.members.member_uav_count, 2))
            )
            self._assert_empty_scheduling_state()
            requests = []
            for dag_offset in range(self.DAGS_PER_SLOT):
                user_id = int(self.rng.integers(self.users.num_users))
                owner_id = int(self.user_member_ids[user_id])
                dag = self._small_dag()
                requests.append(DAGRequest(
                    dag_id=slot_id * self.DAGS_PER_SLOT + dag_offset, dag=dag,
                    owner_member=EntityRef(EntityKind.MEMBER_UAV, owner_id),
                    ground_device=EntityRef(EntityKind.GROUND_DEVICE, user_id)))

            plan = ERS(self.runtime).plan(requests)
            slot_task_keys = self.runtime.register_dags(requests)
            run_online_schedule(self.runtime, plan.member_orders, lambda key: PlacementDecision(
                self._random_execution_node(key.user_id, key.owner_member_id)))
            for order in plan.member_orders.values():
                for previous, current in zip(order, order[1:]):
                    self.assertEqual(self.runtime.trace(current).decision_at,
                                     self.runtime.trace(previous).compute_finish_at)

            slot_end = slot_start + self.max_duration_s
            result = self.environment.end_slot()
            self.assertEqual(result.slot_end, slot_end)
            self.assertEqual(len(result.finished_dags), self.DAGS_PER_SLOT)
            self.assertFalse(result.failed_dags)
            self.assertIsNone(self.environment.runtime)
            self.assertTrue(self.runtime.closed)
            for task_key in slot_task_keys:
                task = self.runtime.trace(task_key)
                self.assertEqual(task.status, TaskStatus.FINISHED)
                self.assertLessEqual(task.compute_finish_at, slot_end)

            self.assertEqual(set(self.runtime.tasks), set(slot_task_keys))
            self.assertEqual(len(self.runtime.dag_runtimes), self.DAGS_PER_SLOT)

    def test_unfinished_dag_does_not_occupy_next_slot_resources(self):
        actions = np.zeros((self.members.member_uav_count, 2))
        self._start_slot(actions)
        ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        owner = EntityRef(EntityKind.MEMBER_UAV, int(self.user_member_ids[0]))
        bs = EntityRef(EntityKind.BS, 0)
        # 人为加大负载，让窗口结束时保留运行中任务与未调度任务。
        overloaded_dag = DAG(
            node_num=3,
            edges=[],
            node_features=np.array([
                [1.0, 2.0 * self.max_duration_s * self.users.config.core_frequency],
                [1.0, 1e7],
                [1e6, 1e7],
            ]),
            edge_features={},
        )
        request = DAGRequest(0, overloaded_dag, owner, ground)
        keys = self.runtime.register_dags([request])
        self.runtime.submit_placements({keys[0]: PlacementDecision(ground)})
        self.runtime.advance_until(self.max_duration_s / 2.0)
        previous_runtime = self.runtime
        self.assertFalse(next(iter(previous_runtime.dag_runtimes.values())).is_finished)
        self.assertTrue(previous_runtime.servers[ground].running)
        self.assertFalse(previous_runtime.servers[ground].queue)
        self.assertTrue(all(previous_runtime.trace(key).status is TaskStatus.UNSCHEDULED for key in keys[1:]))

        result = self.environment.end_slot()
        self.assertEqual(result.failed_dags, ((owner.index, ground.index, 0),))
        self.assertTrue(previous_runtime.closed)
        self.assertTrue(all(task.status is TaskStatus.FAILED for task in previous_runtime.tasks.values()))

        self._start_slot(actions)
        self._assert_empty_scheduling_state()
        self.assertIsNot(self.runtime, previous_runtime)
        # 复用 DAG 标识，确认新时隙不受旧任务或旧调度状态影响。
        request = DAGRequest(0, DAG(1, [], np.array([[1., 1e7]]), {}),
                             EntityRef(EntityKind.MEMBER_UAV, int(self.user_member_ids[0])), ground)
        keys = self.runtime.register_dags([request])
        self.runtime.submit_placements({keys[0]: PlacementDecision(ground)})
        self.environment.end_slot()
        task = self.runtime.trace(keys[0])
        self.assertEqual(task.status, TaskStatus.FINISHED)
        self.assertEqual(task.compute_start_at, self.max_duration_s)
        self.assertAlmostEqual(
            task.compute_finish_at,
            self.max_duration_s + 1e7 / self.users.config.core_frequency,
        )
        self.assertFalse(next(iter(previous_runtime.dag_runtimes.values())).is_finished)

    def _start_slot(self, actions):
        self.environment.begin_slot(self.region, self.users.positions, actions)
        self.runtime = self.environment.runtime
        self.user_member_ids = self.environment.user_member_ids.copy()

    def _assert_empty_scheduling_state(self):
        self.assertEqual(self.runtime.tasks, {})
        self.assertEqual(self.runtime.dag_runtimes, {})
        self.assertEqual(self.runtime.channels, {})
        for server in self.runtime.servers.values():
            self.assertFalse(server.queue)
            self.assertFalse(server.running)

    def _current_uav_positions(self):
        positions = {
            EntityRef(EntityKind.MEMBER_UAV, member_id): self.members.positions[member_id]
            for member_id in range(self.members.member_uav_count)
        }
        positions[EntityRef(EntityKind.BS, 0)] = self.members.positions[self.members.bs_index]
        return positions

    def _random_execution_node(self, user_id, owner_id):
        owner_region_id = self.members.region_ids[owner_id]
        candidates = [EntityRef(EntityKind.GROUND_DEVICE, user_id)]
        candidates.extend(
            [
                EntityRef(EntityKind.MEMBER_UAV, member_id)
                for member_id in range(self.members.member_uav_count)
                if self.members.region_ids[member_id] == owner_region_id
            ]
        )
        candidates.append(EntityRef(EntityKind.BS, 0))
        return candidates[int(self.rng.integers(len(candidates)))]

    def _small_dag(self):
        input_kbit = self.rng.uniform(1.0, 2.0, size=(3, 1))
        cpu_cycles = self.rng.uniform(1e6, 3e6, size=(3, 1))
        return DAG(
            node_num=3,
            edges=[(0, 1), (0, 2)],
            node_features=np.hstack((input_kbit, cpu_cycles)),
            edge_features={(0, 1): 1.0, (0, 2): 1.0},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
