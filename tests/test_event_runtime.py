"""DAG 传输、依赖与计算的离散事件集成测试。"""

import unittest

import numpy as np

from env.communication.channel_model import ChannelModel
from env.contracts import DAGRequest, PlacementDecision
from methods.scheduling import run_online_schedule
from env.runtime.event_runtime import SchedulingRuntime, ServerSpec
from env.types import DirectedChannelKey, EntityKind, EntityRef
from env.workload.dag_generator import DAG


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

    def register(self, dag=None):
        request = DAGRequest(0, dag or self.dag, self.member, self.ground)
        return self.runtime.register_dags([request])

    def test_workload_sizes_are_decimal_kbit(self):
        parent, child = self.register()
        self.runtime.submit_placements({parent: PlacementDecision(self.member)})
        input_hop = DirectedChannelKey(self.ground, self.member)
        self.assertEqual(self.runtime.trace(parent).input_bits, 1000.)
        self.assertAlmostEqual(self.runtime.channel_state(input_hop).active.duration_s,
                               self.runtime.transfer_duration_s(input_hop, 1000.))
        self.runtime.advance_to_decision()
        self.runtime.submit_placements({child: PlacementDecision(self.bs)})
        edge_hop = DirectedChannelKey(self.member, self.bs)
        transfer_id = self.runtime.trace(child).predecessor_records[parent].transfer_ids[0]
        self.assertAlmostEqual(self.runtime.transfer_jobs[transfer_id].duration_s,
                               self.runtime.transfer_duration_s(edge_hop, 1000.))

    def test_relayed_inputs_and_local_predecessor_enable_computation(self):
        keys = self.register()
        run_online_schedule(self.runtime, {self.member: keys}, lambda key: PlacementDecision(self.bs))
        parent, child = (self.runtime.trace(key) for key in keys)
        self.assertEqual(len(parent.input_record.transfer_ids), 2)
        self.assertEqual(child.predecessor_arrival_at[parent.key], child.decision_at)
        self.assertEqual(child.decision_at, parent.compute_finish_at)
        for task in (parent, child):
            self.assertGreaterEqual(task.compute_start_at, task.data_ready_at)
            self.assertEqual(task.status.value, 'finished')

    def test_active_transfer_remains_active_before_its_finish_time(self):
        keys = self.register()
        self.runtime.submit_placements({keys[0]: PlacementDecision(self.member)})
        channel = self.runtime.channel_state(DirectedChannelKey(self.ground, self.member))
        active = channel.active
        self.runtime.advance_until(active.finish_at / 2.)
        self.assertIs(channel.active, active)
        self.assertIsNone(self.runtime.trace(keys[1]).input_record)

    def test_each_decision_uses_post_flight_topology_snapshot(self):
        position = np.array([80., 0., 10.])
        self.runtime.update_entity_positions({self.member: position})
        keys = self.register()
        run_online_schedule(self.runtime, {self.member: keys}, lambda key: PlacementDecision(self.member))
        hop = DirectedChannelKey(self.ground, self.member)
        expected = self.runtime.transfer_duration_s(hop, 1000.)
        for key in keys:
            record = self.runtime.trace(key).input_record
            self.assertAlmostEqual(record.finish_at - record.start_at, expected)

    def test_all_local_dag_finishes_without_transfers_or_compute_energy(self):
        keys = self.register()
        run_online_schedule(self.runtime, {self.member: keys}, lambda key: PlacementDecision(self.ground))
        self.assertFalse(self.runtime.channels)
        for key in keys:
            task = self.runtime.trace(key)
            self.assertEqual(task.status.value, 'finished')
            self.assertEqual(task.input_record.transfer_ids, ())
            self.assertEqual(task.input_record.start_at, task.decision_at)
            self.assertEqual(task.input_record.finish_at, task.decision_at)
            self.assertEqual(task.compute_energy_j, 0.)

    def test_bs_result_returns_to_local_child_through_owner_member(self):
        keys = self.register()
        run_online_schedule(self.runtime, {self.member: keys},
                            lambda key: PlacementDecision(self.bs if key.node_id == 0 else self.ground))
        parent, child = (self.runtime.trace(key) for key in keys)
        self.assertEqual(child.status.value, 'finished')
        self.assertEqual(len(child.predecessor_records[parent.key].transfer_ids), 2)
        self.assertIn(DirectedChannelKey(self.bs, self.member), self.runtime.channels)
        self.assertIn(DirectedChannelKey(self.member, self.ground), self.runtime.channels)


if __name__ == '__main__':
    unittest.main()
