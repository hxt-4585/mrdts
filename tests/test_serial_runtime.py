"""在线卸载的可手算事件场景；策略在测试调用方选择任务与位置。"""

import unittest

import numpy as np

from env.communication.channel_model import ChannelModel
from env.contracts import DAGRequest, PlacementDecision
from env.runtime.event_runtime import SchedulingRuntime, ServerSpec
from env.runtime.task_runtime import TaskStatus
from env.types import DirectedChannelKey, EntityKind, EntityRef
from env.workload.dag_generator import DAG


def make_runtime(duration=30.0, cores=1):
    members = tuple(EntityRef(EntityKind.MEMBER_UAV, i) for i in range(4))
    grounds = tuple(EntityRef(EntityKind.GROUND_DEVICE, i) for i in range(4))
    bs = EntityRef(EntityKind.BS, 0)
    positions = {node: np.array([i * 10., 0., 10. if node in members else 0.])
                 for i, node in enumerate((*members, *grounds, bs))}
    model = ChannelModel()
    bandwidths = {}
    for source in positions:
        for target in positions:
            if source == target:
                continue
            hop = DirectedChannelKey(source, target)
            rate = model.calculate_link(positions[source], positions[target], 1.,
                                        SchedulingRuntime._link_type(hop), 1.).rate_bps
            bandwidths[hop] = 1000. / rate
    runtime = SchedulingRuntime(
        model, positions, {node: ServerSpec((10.,) * cores, .01) for node in positions},
        {node: 1. for node in positions}, bandwidths,
        member_regions={node: 1 for node in members},
        ground_owner_members=dict(zip(grounds, members)), max_duration_s=duration,
    )
    return runtime, members, grounds, bs


def request(member, ground, dag_id=0, inputs=(0.,), cycles=(10.,), edges=(), sizes=()):
    return DAGRequest(dag_id, DAG(len(inputs), list(edges), np.array(list(zip(inputs, cycles))),
                                 dict(zip(edges, sizes))), member, ground)


class TestSerialRuntime(unittest.TestCase):
    def setUp(self):
        self.runtime, self.members, self.grounds, self.bs = make_runtime()

    def submit(self, key, target):
        self.runtime.submit_placements({key: PlacementDecision(target)})

    def test_registration_leaves_all_dags_unscheduled_and_method_chooses_next(self):
        a = request(self.members[0], self.grounds[0], inputs=(0., 0.), cycles=(10., 10.))
        b = request(self.members[0], self.grounds[1], dag_id=1)
        keys = self.runtime.register_dags([a, b])
        self.assertEqual(self.runtime.ready_member_ids, (0,))
        self.assertTrue(all(self.runtime.trace(key).status is TaskStatus.UNSCHEDULED for key in keys))
        self.assertFalse(self.runtime.channels)
        self.submit(b.task_keys[0], self.bs)
        self.assertEqual(self.runtime.active_tasks, {0: b.task_keys[0]})
        self.assertEqual(self.runtime.ready_member_ids, ())
        self.assertEqual(self.runtime.advance_to_decision(), (0,))
        self.assertEqual(self.runtime.now, 1.)
        self.assertIsNone(self.runtime.trace(a.task_keys[0]).execution_node)
        self.submit(a.task_keys[1], self.members[0])
        self.assertEqual(self.runtime.advance_to_decision(), (0,))
        self.submit(a.task_keys[0], self.members[0])
        self.assertEqual(self.runtime.advance_to_decision(), ())
        self.assertEqual(self.runtime.now, 3.)
        self.assertEqual(len(self.runtime.finish_slot().finished_dags), 2)

    def test_asynchronous_completion_wakes_owner_without_blocking_its_server(self):
        a = request(self.members[0], self.grounds[0], inputs=(0., 0.), cycles=(50., 10.))
        b = request(self.members[1], self.grounds[1], inputs=(0., 0.), cycles=(10., 10.))
        self.runtime.register_dags([a, b])
        self.runtime.submit_placements({a.task_keys[0]: PlacementDecision(self.bs),
                                        b.task_keys[0]: PlacementDecision(self.members[0])})
        self.assertEqual(self.runtime.advance_until(20.), (1,))
        self.assertEqual(self.runtime.now, 1.)
        self.assertEqual(self.runtime.active_tasks, {0: a.task_keys[0]})
        self.submit(b.task_keys[1], self.members[0])
        self.assertEqual(self.runtime.advance_to_decision(), (0,))
        self.assertEqual(self.runtime.now, 5.)
        self.assertEqual(self.runtime.trace(b.task_keys[1]).compute_finish_at, 2.)

    def test_input_and_result_start_together_but_shared_hop_is_serial(self):
        dag = request(self.members[0], self.grounds[0], inputs=(0., 2.), cycles=(10., 10.),
                      edges=((0, 1),), sizes=(3.,))
        parent, child = self.runtime.register_dags([dag])
        self.submit(parent, self.members[0])
        self.runtime.advance_to_decision()
        self.assertFalse(self.runtime.trace(child).predecessor_records)
        self.submit(child, self.bs)
        task = self.runtime.trace(child)
        self.assertEqual(task.input_record.start_at, 1.)
        self.assertEqual(task.predecessor_records[parent].start_at, 1.)
        self.runtime.advance_until(3.)
        self.assertIsNone(task.compute_start_at)
        self.runtime.advance_to_decision()
        self.assertAlmostEqual(task.predecessor_arrival_at[parent], 4.)
        self.assertAlmostEqual(task.input_arrival_at, 6.)
        self.assertAlmostEqual(task.compute_start_at, 6.)

    def test_two_results_on_same_channel_serialize(self):
        dag = request(self.members[0], self.grounds[0], inputs=(0., 0., 0.),
                      cycles=(10., 10., 10.), edges=((0, 2), (1, 2)), sizes=(2., 3.))
        first, second, child = self.runtime.register_dags([dag])
        for key in (first, second):
            self.submit(key, self.members[0])
            self.runtime.advance_to_decision()
        self.submit(child, self.bs)
        task = self.runtime.trace(child)
        self.runtime.advance_to_decision()
        self.assertAlmostEqual(task.predecessor_records[first].start_at, 2.)
        self.assertAlmostEqual(task.predecessor_records[first].finish_at, 4.)
        self.assertAlmostEqual(task.predecessor_records[second].start_at, 4.)
        self.assertAlmostEqual(task.predecessor_records[second].finish_at, 7.)

    def test_relay_does_not_reserve_channel_before_source_data_arrives(self):
        slow = request(self.members[0], self.grounds[0], inputs=(5.,))
        fast = request(self.members[1], self.grounds[1], inputs=(0., 0.), cycles=(10., 10.),
                       edges=((0, 1),), sizes=(1.,))
        self.runtime.register_dags([slow, fast])
        self.runtime.submit_placements({slow.task_keys[0]: PlacementDecision(self.bs),
                                        fast.task_keys[0]: PlacementDecision(self.members[0])})
        self.runtime.advance_to_decision()
        self.submit(fast.task_keys[1], self.bs)
        self.runtime.advance_to_decision()
        child = self.runtime.trace(fast.task_keys[1])
        self.assertAlmostEqual(child.predecessor_records[fast.task_keys[0]].start_at, 1.)
        self.assertAlmostEqual(child.compute_finish_at, 3.)
        self.assertAlmostEqual(self.runtime.trace(slow.task_keys[0]).input_arrival_at, 10.)

    def test_waiting_data_does_not_reserve_core_or_fifo_position(self):
        slow = request(self.members[0], self.grounds[0], inputs=(0., 0.), cycles=(10., 10.),
                       edges=((0, 1),), sizes=(5.,))
        fast = request(self.members[1], self.grounds[1], inputs=(0., 0.), cycles=(20., 10.))
        self.runtime.register_dags([slow, fast])
        self.runtime.submit_placements({slow.task_keys[0]: PlacementDecision(self.members[0]),
                                        fast.task_keys[0]: PlacementDecision(self.members[1])})
        self.runtime.advance_to_decision()
        self.submit(slow.task_keys[1], self.bs)
        self.runtime.advance_to_decision()
        self.submit(fast.task_keys[1], self.bs)
        self.assertEqual(self.runtime.trace(fast.task_keys[1]).compute_start_at, 2.)
        self.assertFalse(self.runtime.servers[self.bs].queue)
        self.runtime.advance_to_decision()
        self.assertAlmostEqual(self.runtime.trace(slow.task_keys[1]).compute_start_at, 6.)

    def test_multicore_uses_first_released_core_and_stable_simultaneous_order(self):
        self.runtime, self.members, self.grounds, self.bs = make_runtime(cores=2)
        dags = [request(m, g, cycles=(c,))
                for m, g, c in zip(self.members, self.grounds, (30., 10., 10., 10.))]
        self.runtime.register_dags(reversed(dags))
        self.runtime.submit_placements({d.task_keys[0]: PlacementDecision(self.bs) for d in reversed(dags)})
        self.runtime.advance_to_decision()
        traces = [self.runtime.trace(d.task_keys[0]) for d in dags]
        self.assertEqual([t.compute_start_at for t in traces], [0., 0., 1., 2.])
        self.assertEqual([t.execution_core_id for t in traces], [0, 1, 1, 1])

    def test_deadline_closes_unscheduled_waiting_queued_running_with_partial_energy(self):
        self.runtime, self.members, self.grounds, self.bs = make_runtime(duration=1.)
        dags = [request(self.members[0], self.grounds[0], inputs=(0., 0.), cycles=(20., 10.)),
                request(self.members[1], self.grounds[1]),
                request(self.members[2], self.grounds[2], inputs=(10.,)),
                request(self.members[3], self.grounds[3])]
        self.runtime.register_dags(dags)
        self.runtime.submit_placements({d.task_keys[0]: PlacementDecision(
            self.members[3] if i == 3 else self.bs) for i, d in enumerate(dags)})
        result = self.runtime.finish_slot()
        self.assertEqual(len(result.finished_dags), 1)
        self.assertEqual(len(result.failed_dags), 3)
        self.assertEqual(self.runtime.trace(dags[3].task_keys[0]).compute_finish_at, 1.)
        self.assertIsNone(self.runtime.trace(dags[0].task_keys[1]).decision_at)
        self.assertAlmostEqual(result.compute_energy_j, 20.)
        self.assertAlmostEqual(result.tx_energy_j, 1.)
        self.assertFalse(self.runtime.active_tasks)
        self.assertEqual(self.runtime.ready_member_ids, ())
        self.assertIs(result, self.runtime.finish_slot())

    def test_simultaneous_completions_and_zero_duration_settle_before_observation(self):
        self.runtime, self.members, self.grounds, self.bs = make_runtime(cores=2)
        dags = [request(m, g, inputs=(0., 0.), cycles=(10., 0.), edges=((0, 1),), sizes=(0.,))
                for m, g in zip(self.members[:2], self.grounds[:2])]
        self.runtime.register_dags(dags)
        self.runtime.submit_placements({d.task_keys[0]: PlacementDecision(self.bs) for d in dags})
        self.assertEqual(self.runtime.advance_to_decision(), (0, 1))
        self.assertEqual(self.runtime.now, 1.)
        self.assertFalse(self.runtime.active_tasks)
        self.assertFalse(self.runtime.servers[self.bs].running)
        self.runtime.submit_placements({d.task_keys[1]: PlacementDecision(self.bs) for d in dags})
        self.assertFalse(self.runtime.active_tasks)
        self.assertEqual(self.runtime.ready_member_ids, ())
        self.assertTrue(all(self.runtime.trace(d.task_keys[1]).compute_finish_at == 1. for d in dags))

    def test_already_queued_work_precedes_new_decision_at_same_timestamp(self):
        a = request(self.members[0], self.grounds[0], inputs=(0., 0.), cycles=(10., 10.))
        b = request(self.members[1], self.grounds[1])
        c = request(self.members[2], self.grounds[2])
        self.runtime.register_dags([a, b, c])
        self.runtime.submit_placements({a.task_keys[0]: PlacementDecision(self.members[0]),
                                        b.task_keys[0]: PlacementDecision(self.bs),
                                        c.task_keys[0]: PlacementDecision(self.bs)})
        self.assertEqual(self.runtime.advance_to_decision(), (0,))
        self.assertEqual(self.runtime.trace(c.task_keys[0]).compute_start_at, 1.)
        self.submit(a.task_keys[1], self.bs)
        self.runtime.advance_to_decision()
        self.assertEqual(self.runtime.trace(a.task_keys[1]).compute_start_at, 2.)


if __name__ == '__main__':
    unittest.main()
