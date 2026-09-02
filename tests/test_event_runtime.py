"""DAG 传输、依赖与计算的离散事件集成测试。"""

import unittest

import numpy as np

from env.channel_model import ChannelModel
from env.channel_queue import DirectedChannelKey, EntityKind, EntityRef
from env.dag_generator import DAG
from env.event_runtime import SchedulingRuntime, ServerSpec
from methods.contracts import PlacementDecision


class TestSchedulingRuntime(unittest.TestCase):
    def setUp(self):
        self.ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        self.member = EntityRef(EntityKind.MEMBER_UAV, 0)
        self.other_member = EntityRef(EntityKind.MEMBER_UAV, 1)
        self.bs = EntityRef(EntityKind.BS, 0)
        positions = {
            self.ground: np.array([0.0, 0.0, 0.0]),
            self.member: np.array([0.0, 0.0, 10.0]),
            self.other_member: np.array([30.0, 0.0, 10.0]),
            self.bs: np.array([50.0, 0.0, 25.0]),
        }
        servers = {
            self.ground: ServerSpec((1e9,), 0.0),
            self.member: ServerSpec((1e9,), 1e-28),
            self.other_member: ServerSpec((1e9,), 1e-28),
            self.bs: ServerSpec((1e9,), 1e-28),
        }
        self.runtime = SchedulingRuntime(
            channel_model=ChannelModel(),
            entity_positions=positions,
            servers=servers,
            transmit_powers={
                self.ground: 1.0,
                self.member: 1.0,
                self.other_member: 1.0,
                self.bs: 1.0,
            },
            directed_bandwidth_hz={
                DirectedChannelKey(self.ground, self.member): 1e9,
                DirectedChannelKey(self.member, self.bs): 1e9,
            },
            member_regions={self.member: 1, self.other_member: 1},
            ground_owner_members={self.ground: self.member},
        )
        self.dag = DAG(
            node_num=2,
            edges=[(0, 1)],
            node_features=np.array([[1.0, 1e7], [1.0, 1e7]]),
            edge_features={(0, 1): 1.0},
        )

    def test_relayed_inputs_and_local_predecessor_enable_fifo_computation(self):
        keys = self.runtime.submit_dag(
            dag_id=7,
            dag=self.dag,
            owner_member=self.member,
            ground_device=self.ground,
            placements={
                0: PlacementDecision(self.bs, ers_seq=0),
                1: PlacementDecision(self.bs, ers_seq=1),
            },
            epoch_start=0.0,
        )

        self.assertIsNotNone(
            self.runtime.channel_state(DirectedChannelKey(self.ground, self.member)).active
        )
        self.runtime.advance_until(10.0)
        parent = self.runtime.trace(keys[0])
        child = self.runtime.trace(keys[1])

        self.assertEqual(len(parent.input_record.transfer_ids), 2)
        self.assertIsNotNone(parent.input_record.finish_at)
        self.assertEqual(child.predecessor_arrival_at[keys[0]], parent.compute_finish_at)
        self.assertGreaterEqual(parent.compute_start_at, parent.data_ready_at)
        self.assertGreaterEqual(child.compute_start_at, child.data_ready_at)
        self.assertEqual(parent.status.value, "finished")
        self.assertEqual(child.status.value, "finished")

    def test_active_transfer_remains_active_before_its_finish_time(self):
        """当前时隙内推进到传输中途时，不应提前完成活动传输。"""
        self.runtime.submit_dag(
            dag_id=8,
            dag=self.dag,
            owner_member=self.member,
            ground_device=self.ground,
            placements={
                0: PlacementDecision(self.member, ers_seq=0),
                1: PlacementDecision(self.member, ers_seq=1),
            },
            epoch_start=0.0,
        )
        key = DirectedChannelKey(self.ground, self.member)
        active = self.runtime.channel_state(key).active
        active_id = active.transfer_id

        self.runtime.advance_until(active.finish_at / 2.0)

        self.assertEqual(self.runtime.channel_state(key).active.transfer_id, active_id)

    def test_queued_transfer_keeps_the_post_flight_topology_duration(self):
        post_flight_position = np.array([80.0, 0.0, 10.0])
        self.runtime.update_entity_positions({self.member: post_flight_position})
        self.runtime.submit_dag(
            dag_id=9,
            dag=self.dag,
            owner_member=self.member,
            ground_device=self.ground,
            placements={
                0: PlacementDecision(self.member, ers_seq=0),
                1: PlacementDecision(self.member, ers_seq=1),
            },
            epoch_start=0.0,
        )
        key = DirectedChannelKey(self.ground, self.member)
        first = self.runtime.channel_state(key).active
        self.runtime.advance_until(first.finish_at / 2.0)
        self.runtime.update_entity_positions({self.member: np.array([150.0, 0.0, 10.0])})

        self.runtime.advance_until(first.finish_at)

        second = self.runtime.channel_state(key).active
        expected_duration = self.runtime.channel_model.transmission_delay_s(
            data_size_bits=8.0 * 1024.0,
            transmitter=self.runtime.entity_positions[self.ground],
            receiver=post_flight_position,
            transmit_power_w=1.0,
            link_type="ground_to_air",
            bandwidth_hz=1e9,
        )
        self.assertAlmostEqual(second.duration_s, expected_duration)

    def test_rejects_member_placement_outside_the_owner_current_region(self):
        self.runtime.update_topology(
            member_regions={self.member: 1, self.other_member: 2},
            ground_owner_members={self.ground: self.member},
        )

        with self.assertRaisesRegex(ValueError, "同一区域"):
            self.runtime.submit_dag(
                dag_id=10,
                dag=self.dag,
                owner_member=self.member,
                ground_device=self.ground,
                placements={
                    0: PlacementDecision(self.other_member, ers_seq=0),
                    1: PlacementDecision(self.other_member, ers_seq=1),
                },
                epoch_start=0.0,
            )

    def test_rejects_placement_on_another_ground_device(self):
        other_ground = EntityRef(EntityKind.GROUND_DEVICE, 1)

        with self.assertRaisesRegex(ValueError, "自身地面设备"):
            self.runtime.submit_dag(
                dag_id=11,
                dag=self.dag,
                owner_member=self.member,
                ground_device=self.ground,
                placements={
                    0: PlacementDecision(other_ground, ers_seq=0),
                    1: PlacementDecision(self.member, ers_seq=1),
                },
                epoch_start=0.0,
            )

    def test_requires_topology_before_accepting_placements(self):
        self.runtime.member_regions = None
        self.runtime.ground_owner_members = None

        with self.assertRaisesRegex(RuntimeError, "拓扑"):
            self.runtime.submit_dag(
                dag_id=14,
                dag=self.dag,
                owner_member=self.member,
                ground_device=self.ground,
                placements={
                    0: PlacementDecision(self.member, ers_seq=0),
                    1: PlacementDecision(self.bs, ers_seq=1),
                },
                epoch_start=0.0,
            )

    def test_rejects_any_bs_other_than_the_single_global_bs(self):
        other_bs = EntityRef(EntityKind.BS, 1)

        with self.assertRaisesRegex(ValueError, "全局 BS"):
            self.runtime.submit_dag(
                dag_id=15,
                dag=self.dag,
                owner_member=self.member,
                ground_device=self.ground,
                placements={
                    0: PlacementDecision(other_bs, ers_seq=0),
                    1: PlacementDecision(self.member, ers_seq=1),
                },
                epoch_start=0.0,
            )

    def test_rejects_ground_server_that_is_not_single_core_and_zero_energy(self):
        for invalid_spec in (ServerSpec((1e9, 1e9), 0.0), ServerSpec((1e9,), 1e-28)):
            with self.subTest(invalid_spec=invalid_spec):
                with self.assertRaisesRegex(ValueError, "地面设备.*单核.*零计算能耗"):
                    SchedulingRuntime(
                        channel_model=ChannelModel(),
                        entity_positions={
                            self.ground: np.array([0.0, 0.0, 0.0]),
                            self.member: np.array([0.0, 0.0, 10.0]),
                            self.bs: np.array([50.0, 0.0, 25.0]),
                        },
                        servers={
                            self.ground: invalid_spec,
                            self.member: ServerSpec((1e9,), 1e-28),
                            self.bs: ServerSpec((1e9,), 1e-28),
                        },
                        transmit_powers={
                            self.ground: 1.0,
                            self.member: 1.0,
                            self.bs: 1.0,
                        },
                        member_regions={self.member: 1},
                        ground_owner_members={self.ground: self.member},
                    )

    def test_all_local_dag_finishes_without_transfers_or_compute_energy(self):
        keys = self.runtime.submit_dag(
            dag_id=12,
            dag=self.dag,
            owner_member=self.member,
            ground_device=self.ground,
            placements={
                0: PlacementDecision(self.ground, ers_seq=0),
                1: PlacementDecision(self.ground, ers_seq=1),
            },
            epoch_start=0.0,
        )

        self.runtime.advance_until(1.0)

        self.assertEqual(self.runtime.channels, {})
        for key in keys:
            task = self.runtime.trace(key)
            self.assertEqual(task.status.value, "finished")
            self.assertEqual(task.input_record.transfer_ids, ())
            self.assertEqual(task.input_record.start_at, 0.0)
            self.assertEqual(task.input_record.finish_at, 0.0)
            self.assertEqual(task.compute_energy_j, 0.0)

    def test_bs_result_returns_to_local_child_through_owner_member(self):
        keys = self.runtime.submit_dag(
            dag_id=13,
            dag=self.dag,
            owner_member=self.member,
            ground_device=self.ground,
            placements={
                0: PlacementDecision(self.bs, ers_seq=0),
                1: PlacementDecision(self.ground, ers_seq=1),
            },
            epoch_start=0.0,
        )

        self.runtime.advance_until(10.0)

        parent = self.runtime.trace(keys[0])
        child = self.runtime.trace(keys[1])
        self.assertEqual(parent.status.value, "finished")
        self.assertEqual(child.status.value, "finished")
        self.assertEqual(len(child.predecessor_records[keys[0]].transfer_ids), 2)
        self.assertIn(DirectedChannelKey(self.bs, self.member), self.runtime.channels)
        self.assertIn(DirectedChannelKey(self.member, self.ground), self.runtime.channels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
