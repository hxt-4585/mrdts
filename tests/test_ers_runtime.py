"""统一 ERS 顺序必须进入真实传输和多核执行，批次失败不得留下半成品。"""

import unittest

import numpy as np

from env.channel_queue import DirectedChannelKey
from env.dag_generator import DAG
from methods.contracts import DAGRequest, PlacementDecision
from methods.ers import ERS
from tests.test_ers import make_dag, make_runtime


class TestERSRuntime(unittest.TestCase):
    def test_interleaved_dags_start_transfers_in_plan_order(self):
        traces = []
        for reverse in (False, True):
            runtime, ground, member, _ = make_runtime()
            requests = [DAGRequest(0, make_dag((4e8, 1e8)), member, ground),
                        DAGRequest(1, make_dag((2e8, 2e8)), member, ground)]
            if reverse:
                requests.reverse()
            plan = ERS(runtime).plan(requests)
            placements = {key: PlacementDecision(member, seq) for seq, key in enumerate(plan.order)}
            keys = runtime.submit_dags(requests, placements, runtime.now)
            self.assertEqual(set(keys), set(plan.order))
            channel = runtime.channel_state(DirectedChannelKey(ground, member))
            self.assertEqual(channel.active.ers_seq, 0)
            self.assertEqual([job.ers_seq for job in channel.queued], [1, 2, 3])
            self.assertEqual(runtime.trace(plan.order[0]).input_record.start_at, 0.)
            self.assertIsNone(runtime.trace(plan.order[1]).input_record.start_at)
            result = runtime.finish_slot()
            self.assertEqual(len(result.finished_dags), 2)
            rows = [(key, runtime.trace(key).input_arrival_at, runtime.trace(key).compute_start_at)
                    for key in plan.order]
            for index, key in enumerate(plan.order):
                self.assertEqual(runtime.trace(key).ers_seq, index)
                self.assertAlmostEqual(runtime.trace(key).input_arrival_at, .05 * (index + 1))
            self.assertEqual(sorted(plan.order, key=lambda key: runtime.trace(key).compute_start_at), list(plan.order))
            traces.append(rows)
        self.assertEqual(traces[0], traces[1])

    def test_local_compute_does_not_start_before_whole_batch_is_ranked(self):
        runtime, ground, member, _ = make_runtime()
        low = DAGRequest(0, make_dag((1e7,)), member, ground)
        high = DAGRequest(1, make_dag((3e7,)), member, ground)
        plan = ERS(runtime).plan([low, high])
        self.assertEqual(plan.order[0], high.task_keys[0])
        runtime.submit_dags([low, high], {key: PlacementDecision(ground, seq)
                                        for seq, key in enumerate(plan.order)}, runtime.now)
        self.assertEqual(runtime.trace(high.task_keys[0]).compute_start_at, 0.)
        self.assertIsNone(runtime.trace(low.task_keys[0]).compute_start_at)
        runtime.finish_slot()
        self.assertAlmostEqual(runtime.trace(low.task_keys[0]).compute_start_at, .03)

    def test_actual_input_and_dependency_use_the_same_two_hop_costs(self):
        runtime, ground, member, bs = make_runtime(asymmetric=True)
        request = DAGRequest(0, make_dag(), member, ground)
        plan = ERS(runtime).plan([request])
        keys = request.task_keys
        runtime.submit_dags([request], {keys[0]: PlacementDecision(ground, 0),
                                        keys[1]: PlacementDecision(bs, 1)}, runtime.now)
        runtime.finish_slot()
        parent, child = (runtime.trace(key) for key in keys)
        self.assertEqual(len(child.input_record.transfer_ids), 2)
        self.assertEqual(len(child.predecessor_records[keys[0]].transfer_ids), 2)
        self.assertAlmostEqual(child.input_arrival_at, .06)
        self.assertAlmostEqual(child.predecessor_arrival_at[keys[0]] - parent.compute_finish_at, .06)
        self.assertAlmostEqual(plan.costs[request.key].path_seconds_per_bit[ground, bs] * 1e6, .06)
        self.assertAlmostEqual(child.compute_start_at, .16)

    def test_invalid_later_request_does_not_mutate_runtime(self):
        for failure in ("placement", "features", "cycle", "duplicate", "rank", "missing", "ties", "route"):
            with self.subTest(failure=failure):
                runtime, ground, member, bs = make_runtime()
                first = DAGRequest(0, make_dag((1e7,)), member, ground)
                second = DAGRequest(1, make_dag(), member, ground)
                requests = [first, second]
                placements = {first.task_keys[0]: PlacementDecision(ground, 0),
                              second.task_keys[0]: PlacementDecision(member, 1),
                              second.task_keys[1]: PlacementDecision(bs, 2)}
                if failure == "placement":
                    from env.channel_queue import EntityKind, EntityRef
                    placements[second.task_keys[1]] = PlacementDecision(EntityRef(EntityKind.BS, 99), 2)
                elif failure == "features":
                    second.dag.node_features[1, 0] = np.nan
                elif failure == "cycle":
                    second.dag.edges.append((1, 0))
                    second.dag.edge_features[1, 0] = 1.
                elif failure == "duplicate":
                    requests.append(first)
                elif failure == "rank":
                    placements[second.task_keys[0]] = PlacementDecision(member, 3)
                elif failure == "missing":
                    del placements[second.task_keys[1]]
                elif failure == "ties":
                    placements[second.task_keys[1]] = PlacementDecision(bs, 1)
                elif failure == "route":
                    runtime.update_entity_positions({bs: runtime.entity_positions[member]})
                with self.assertRaises(ValueError):
                    runtime.submit_dags(requests, placements, runtime.now)
                self.assertFalse(runtime.tasks)
                self.assertFalse(runtime.dag_runtimes)
                self.assertFalse(runtime.channels)
                self.assertTrue(all(not s.queue and not s.running for s in runtime.servers.values()))
                keys = runtime.submit_dag(0, first.dag, member, ground,
                                          {0: PlacementDecision(ground, 0)}, runtime.now)
                self.assertEqual(runtime.trace(keys[0]).ers_seq, 0)

    def test_later_batch_appends_without_reordering_running_work(self):
        runtime, ground, member, _ = make_runtime()
        first = DAGRequest(0, make_dag((1e7,)), member, ground)
        second = DAGRequest(1, make_dag((3e7,)), member, ground)
        runtime.submit_dags([first], {first.task_keys[0]: PlacementDecision(ground, 50)}, runtime.now)
        runtime.submit_dags([second], {second.task_keys[0]: PlacementDecision(ground, 0)}, runtime.now)
        self.assertEqual(runtime.trace(first.task_keys[0]).ers_seq, 0)
        self.assertEqual(runtime.trace(second.task_keys[0]).ers_seq, 1)
        runtime.finish_slot()
        self.assertAlmostEqual(runtime.trace(second.task_keys[0]).compute_start_at, .01)

    def test_zero_payload_keeps_dependencies_without_transfer_jobs(self):
        runtime, ground, member, bs = make_runtime()
        request = DAGRequest(0, make_dag(input_kb=0., edge_kb=0.), member, ground)
        keys = request.task_keys
        runtime.submit_dags([request], {keys[0]: PlacementDecision(ground, 0),
                                        keys[1]: PlacementDecision(bs, 1)}, runtime.now)
        self.assertFalse(runtime.channels)
        runtime.finish_slot()
        self.assertAlmostEqual(runtime.trace(keys[1]).compute_start_at, .1)


if __name__ == "__main__":
    unittest.main()
