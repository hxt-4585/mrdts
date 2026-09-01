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

    def test_active_transfer_survives_a_slot_boundary(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
