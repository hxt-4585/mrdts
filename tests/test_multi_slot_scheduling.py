"""十个时隙的随机飞行与随机卸载集成测试。"""

from pathlib import Path
import tomllib
import unittest

import numpy as np

from env.channel_model import ChannelModel
from env.channel_queue import EntityKind, EntityRef
from env.dag_generator import DAG
from env.environment import Environment
from env.task_runtime import TaskStatus
from env.uav import MasterUAV, MemberUAV
from env.region import Region
from env.user import User
from methods.contracts import PlacementDecision
from methods.heft_priority import HEFTPriority


class TestMultiSlotScheduling(unittest.TestCase):
    """随机策略不应破坏独立信道、依赖与时隙 deadline 约束。"""

    SLOT_COUNT = 10
    DAGS_PER_SLOT = 2

    def setUp(self):
        self.region = Region()
        self.region.generate()
        self.users = User()
        self.users.generate_from_region(self.region)
        masters = MasterUAV()
        masters.generate_from_region(self.region)
        self.members = MemberUAV()
        self.members.generate_from_region_and_user(self.region, self.users, masters)
        self.environment = Environment(self.members, ChannelModel(), user_transmit_power=0.1)
        self.user_member_ids = np.array(
            [
                self.environment.member_ids_in_region(region_id)[0]
                for region_id in self.users.region_ids
            ],
            dtype=int,
        )
        self.runtime = self.environment.create_scheduling_runtime(
            self.users.positions, self.user_member_ids
        )
        with (Path(__file__).parents[1] / "config" / "scheduling.toml").open("rb") as file:
            self.max_duration_s = tomllib.load(file)["scheduling"]["max_duration_s"]
        self.rng = np.random.default_rng(20260901)

    def test_random_flight_and_random_placement_finish_two_dags_per_slot_before_deadline(self):
        for slot_id in range(self.SLOT_COUNT):
            slot_start = slot_id * self.max_duration_s
            self.members.apply_flight_actions(
                self.rng.uniform(-1.0, 1.0, size=(self.members.member_uav_count, 2))
            )
            self.user_member_ids = self.environment.refresh_slot_topology(
                self.region, self.users.positions, self.runtime
            )
            slot_task_keys = []
            for dag_offset in range(self.DAGS_PER_SLOT):
                user_id = int(self.rng.integers(self.users.num_users))
                owner_id = int(self.user_member_ids[user_id])
                dag = self._small_dag()
                order = HEFTPriority.order(
                    dag,
                    average_compute_s={0: 0.0003, 1: 0.0004, 2: 0.0004},
                    average_edge_comm_s={(0, 1): 0.0001, (0, 2): 0.0001},
                )
                placements = {
                    node_id: PlacementDecision(
                        execution_node=self._random_execution_node(owner_id),
                        ers_seq=order.index(node_id),
                    )
                    for node_id in range(dag.node_num)
                }
                slot_task_keys.extend(
                    self.runtime.submit_dag(
                        dag_id=slot_id * self.DAGS_PER_SLOT + dag_offset,
                        dag=dag,
                        owner_member=EntityRef(EntityKind.MEMBER_UAV, owner_id),
                        ground_device=EntityRef(EntityKind.GROUND_DEVICE, user_id),
                        placements=placements,
                        epoch_start=slot_start,
                    )
                )

            slot_end = slot_start + self.max_duration_s
            self.runtime.advance_until(slot_end)
            for task_key in slot_task_keys:
                task = self.runtime.trace(task_key)
                self.assertEqual(task.status, TaskStatus.FINISHED)
                self.assertLessEqual(task.compute_finish_at, slot_end)

    def _current_uav_positions(self):
        positions = {
            EntityRef(EntityKind.MEMBER_UAV, member_id): self.members.positions[member_id]
            for member_id in range(self.members.member_uav_count)
        }
        positions[EntityRef(EntityKind.BS, 0)] = self.members.positions[self.members.bs_index]
        return positions

    def _random_execution_node(self, owner_id):
        owner_region_id = self.members.region_ids[owner_id]
        candidates = [
            EntityRef(EntityKind.MEMBER_UAV, member_id)
            for member_id in range(self.members.member_uav_count)
            if self.members.region_ids[member_id] == owner_region_id
        ]
        candidates.append(EntityRef(EntityKind.BS, 0))
        return candidates[int(self.rng.integers(len(candidates)))]

    def _small_dag(self):
        input_kb = self.rng.uniform(1.0, 2.0, size=(3, 1))
        cpu_cycles = self.rng.uniform(1e6, 3e6, size=(3, 1))
        return DAG(
            node_num=3,
            edges=[(0, 1), (0, 2)],
            node_features=np.hstack((input_kb, cpu_cycles)),
            edge_features={(0, 1): 1.0, (0, 2): 1.0},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
