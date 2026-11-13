"""Hand-calculated schedules for a complete placement plan."""

import unittest

import numpy as np

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.runtime.dag_runtime import topological_order
from scripts.ers import ERS
from tests import test_mrdts_env
from tests.test_serial_runtime import make_runtime, request


class TestParallelRuntime(unittest.TestCase):
    def test_zero_chain_releases_higher_priority_edge_before_uplink_dispatch(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 0., 0., 1.), cycles=(0., 0., 10., 10.),
                      edges=((0, 1), (1, 2)), sizes=(1., 1.))
        root, parent, b, a = runtime.register_dags([dag])
        runtime.submit_placements({root: PlacementDecision(grounds[0]), parent: PlacementDecision(grounds[0]),
                                   b: PlacementDecision(members[0]), a: PlacementDecision(members[0])})
        self.assertEqual(runtime.pop_member_rewards(), {})
        runtime.finish_slot()
        self.assertEqual(runtime.trace(b).predecessor_records[parent].start_at, 0.)
        self.assertEqual(runtime.trace(a).input_record.start_at, 1.)

    def test_zero_parent_releases_higher_priority_compute_on_another_server(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 0., 0.), cycles=(0., 10., 10.),
                      edges=((0, 1),), sizes=(0.,))
        parent, b, a = runtime.register_dags([dag])
        runtime.submit_placements({parent: PlacementDecision(grounds[0]), b: PlacementDecision(members[0]),
                                   a: PlacementDecision(members[0])})
        runtime.finish_slot()
        self.assertEqual([runtime.trace(k).compute_start_at for k in (b, a)], [0., 1.])

    def test_zero_parent_waits_for_higher_priority_work_on_its_core(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 0., 0., 1.), cycles=(10., 0., 10., 10.),
                      edges=((1, 2),), sizes=(1.,))
        x, parent, b, a = runtime.register_dags([dag])
        runtime.submit_placements({x: PlacementDecision(grounds[0]), parent: PlacementDecision(grounds[0]),
                                   b: PlacementDecision(members[0]), a: PlacementDecision(members[0])})
        runtime.finish_slot()
        self.assertEqual(runtime.trace(parent).compute_start_at, 1.)
        self.assertEqual(runtime.trace(a).input_record.start_at, 0.)
        self.assertEqual(runtime.trace(b).predecessor_records[parent].start_at, 1.)

    def test_zero_parent_edge_released_at_uplink_handoff_follows_waiting_input(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(1., 0., 0., 0., 1.), cycles=(10., 10., 0., 10., 10.),
                      edges=((1, 2), (2, 3)), sizes=(1., 1.))
        x, root, parent, b, a = runtime.register_dags([dag])
        runtime.submit_placements({x: PlacementDecision(members[0]), root: PlacementDecision(grounds[0]),
                                   parent: PlacementDecision(grounds[0]), b: PlacementDecision(members[0]),
                                   a: PlacementDecision(members[0])})
        runtime.finish_slot()
        self.assertEqual(runtime.trace(parent).compute_finish_at, 1.)
        self.assertEqual(runtime.trace(a).input_record.start_at, 1.)
        self.assertEqual(runtime.trace(b).predecessor_records[parent].start_at, 2.)

    def test_released_core_prefers_earlier_ready_task_despite_later_placement(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 1., 0.), cycles=(100., 10., 10.))
        x, b, a = runtime.register_dags([dag])
        runtime.submit_placements({key: PlacementDecision(members[0]) for key in (x, b, a)})
        runtime.finish_slot()
        for key, expected in ((x, (0., 0., 10.)), (a, (0., 10., 11.)), (b, (1., 11., 12.))):
            task = runtime.trace(key)
            self.assertEqual((task.data_ready_at, task.compute_start_at, task.compute_finish_at), expected)

    def test_released_uplink_prefers_earlier_ready_input_over_later_edge(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(10., 0., 1., 0.), cycles=(10.,) * 4,
                      edges=((3, 1),), sizes=(1.,))
        x, b, a, parent = runtime.register_dags([dag])
        runtime.submit_placements({parent: PlacementDecision(grounds[0]),
                                   **{key: PlacementDecision(members[0]) for key in (x, b, a)}})
        runtime.finish_slot()
        records = (runtime.trace(x).input_record, runtime.trace(a).input_record,
                   runtime.trace(b).predecessor_records[parent])
        for record, expected in zip(records, ((0., 0., 10.), (0., 10., 11.), (1., 11., 12.))):
            hop = runtime.transfer_jobs[record.transfer_ids[0]]
            self.assertEqual((hop.source_ready_at, hop.start_at, hop.finish_at), expected)

    def test_shared_uplink_transmits_raw_inputs_in_ers_order(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(1., 1., 1.), cycles=(20., 10., 30.))
        a, b, c = runtime.register_dags([dag])
        order = ERS(runtime).plan([dag]).order
        self.assertEqual(order, (c, a, b))
        runtime.submit_placements({key: PlacementDecision(members[0]) for key in order})
        runtime.finish_slot()
        intervals = [(runtime.trace(key).input_record.start_at, runtime.trace(key).input_arrival_at)
                     for key in order]
        self.assertEqual(intervals, [(0., 1.), (1., 2.), (2., 3.)])

    def test_simultaneous_edge_transfers_follow_children_in_submitted_order(self):
        runtime, members, grounds, bs = make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 0., 0.), cycles=(10., 10., 10.),
                      edges=((0, 1), (0, 2)), sizes=(1., 1.))
        parent, a, b = runtime.register_dags([dag])
        runtime.submit_placements({parent: PlacementDecision(members[0]),
                                   b: PlacementDecision(bs), a: PlacementDecision(bs)})
        runtime.finish_slot()
        self.assertEqual([runtime.trace(child).predecessor_records[parent].start_at
                          for child in (b, a)], [1., 2.])

    def test_simultaneously_ready_tasks_take_single_core_in_submitted_order(self):
        runtime, members, grounds, _ = make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 0.), cycles=(10., 10.))
        a, b = runtime.register_dags([dag])
        runtime.submit_placements({b: PlacementDecision(members[0]), a: PlacementDecision(members[0])})
        runtime.finish_slot()
        self.assertEqual([runtime.trace(key).compute_start_at for key in (b, a)], [0., 1.])

    def test_same_owner_parallel_roots_and_per_edge_release(self):
        runtime, members, grounds, bs = make_runtime(cores=2)
        dag = request(members[0], grounds[0], inputs=(0., 0., 1.),
                      cycles=(20., 50., 10.), edges=((0, 2), (1, 2)), sizes=(1., 1.))
        a, b, child = runtime.register_dags([dag])
        runtime.submit_placements({child: PlacementDecision(members[0]),
                                   b: PlacementDecision(bs), a: PlacementDecision(bs)})
        result = runtime.finish_slot()
        first, second, last = map(runtime.trace, (a, b, child))
        self.assertEqual((first.compute_start_at, second.compute_start_at), (0., 0.))
        self.assertEqual(last.input_arrival_at, 1.)
        self.assertEqual([last.predecessor_records[p].start_at for p in (a, b)], [2., 5.])
        self.assertEqual((last.compute_start_at, result.dags[0].completion_time), (6., 7.))
        self.assertEqual(set(runtime.pop_member_rewards()), {a, b, child})

    def test_automatic_local_dag_in_empty_plan(self):
        runtime, _, grounds, _ = make_runtime()
        dag = request(None, grounds[0], inputs=(0., 1.), cycles=(0., 20.),
                      edges=((0, 1),), sizes=(1.,))
        parent, child = runtime.register_dags([dag])
        self.assertIsNone(runtime.trace(parent).compute_start_at)
        runtime.submit_placements({})
        self.assertEqual(runtime.finish_slot().dags[0].completion_time, 2.)
        self.assertEqual(runtime.trace(child).predecessor_arrival_at[parent], 0.)
        self.assertFalse(runtime.transfer_jobs)

    def test_public_window_with_a_method_owned_plan(self):
        env = MRDTSEnv(test_mrdts_env.TestMRDTSEnv.settings())
        env.generate_scene()
        env.reset(epoch_index=0)
        begin = env.begin_step(np.zeros((env.members.member_uav_count, 2)))
        placements = {}
        for dag in begin.requests:
            if dag.owner_member is None:
                continue
            keys = dag.task_keys
            for node in topological_order(dag.dag):
                observation = env.observe_member(keys[node], placements)
                incoming = sum(dag.dag.edge_features[parent, node]
                               for parent, child in dag.dag.edges if child == node)
                config = env.settings["dag"]
                scale = (config.n - 1) * config.intermediate_data_range[1]
                self.assertAlmostEqual(observation[2 + dag.owner_member.index], incoming / scale)
                placements[keys[node]] = PlacementDecision(dag.owner_member)
        self.assertTrue(placements)
        env.submit_schedule(placements)
        result = env.end_step()
        self.assertEqual(set(result.member_rewards), set(placements))
        self.assertTrue(all(d.completion_time is not None for d in result.slot_result.dags))
        self.assertTrue(np.isfinite(result.master_reward))
