"""DAG 传输、依赖与计算的离散事件集成测试。"""

import unittest

import numpy as np

from env.communication.channel_model import ChannelModel
from env.contracts import DAGRequest, PlacementDecision
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
            member_regions={self.member: 1, self.other_member: 1},
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
        self.runtime.submit_placements({parent: PlacementDecision(self.member), child: PlacementDecision(self.bs)})
        input_hop = DirectedChannelKey(self.ground, self.member)
        self.assertEqual(self.runtime.trace(parent).input_bits, 1000.)
        self.assertAlmostEqual(self.runtime.channels[input_hop].active.predicted_finish_at - self.runtime.now,
                               self.runtime.transfer_duration_s(input_hop, 1000.))
        self.runtime.finish_slot()
        transfer_id = self.runtime.trace(child).predecessor_records[parent].transfer_ids[0]
        self.assertEqual(self.runtime.transfer_jobs[transfer_id].payload_bits, 1000.)
        self.assertGreaterEqual(self.runtime.transfer_jobs[transfer_id].start_at,
                                self.runtime.trace(parent).compute_finish_at)

    def test_relayed_inputs_and_local_predecessor_enable_computation(self):
        keys = self.register()
        self.runtime.submit_placements(dict.fromkeys(keys, PlacementDecision(self.bs)))
        self.runtime.finish_slot()
        parent, child = (self.runtime.trace(key) for key in keys)
        self.assertEqual(len(parent.input_record.transfer_ids), 2)
        self.assertEqual(child.predecessor_arrival_at[parent.key], parent.compute_finish_at)
        self.assertEqual(child.decision_at, parent.decision_at)
        for task in (parent, child):
            self.assertGreaterEqual(task.compute_start_at, task.data_ready_at)
            self.assertEqual(task.status.value, 'finished')

    def test_active_transfer_remains_active_before_its_finish_time(self):
        keys = self.register()
        self.runtime.submit_placements(dict.fromkeys(keys, PlacementDecision(self.member)))
        channel = self.runtime.channels[DirectedChannelKey(self.ground, self.member)]
        active = channel.active
        self.runtime.advance_until(active.predicted_finish_at / 2.)
        self.assertIs(channel.active, active)
        self.assertIsNone(self.runtime.trace(keys[1]).input_record.start_at)

    def test_all_local_dag_finishes_without_transfers_or_compute_energy(self):
        keys = self.register()
        self.runtime.submit_placements(dict.fromkeys(keys, PlacementDecision(self.ground)))
        self.runtime.finish_slot()
        self.assertFalse(self.runtime.transfer_jobs)
        for key in keys:
            task = self.runtime.trace(key)
            self.assertEqual(task.status.value, 'finished')
            self.assertEqual(task.input_record.transfer_ids, ())
            self.assertEqual(task.input_record.start_at, task.decision_at)
            self.assertEqual(task.input_record.finish_at, task.decision_at)
            self.assertEqual(task.compute_energy_j, 0.)

    def test_bs_result_returns_to_local_child_through_owner_member(self):
        keys = self.register()
        self.runtime.submit_placements({keys[0]: PlacementDecision(self.bs), keys[1]: PlacementDecision(self.ground)})
        self.runtime.finish_slot()
        parent, child = (self.runtime.trace(key) for key in keys)
        self.assertEqual(child.status.value, 'finished')
        self.assertEqual(len(child.predecessor_records[parent.key].transfer_ids), 2)
        first, second = (self.runtime.transfer_jobs[tid] for tid in child.predecessor_records[parent.key].transfer_ids)
        self.assertEqual(first.start_at, parent.compute_finish_at)
        self.assertEqual(second.start_at, first.finish_at)
        self.assertEqual(child.compute_start_at, second.finish_at)


if __name__ == '__main__':
    unittest.main()
