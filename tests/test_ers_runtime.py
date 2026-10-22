"""ERS 属于 method；环境按调用方的 Member 序列接受在线动作。"""

import unittest

from env.contracts import DAGRequest, PlacementDecision
from methods.random.ers import ERS
from methods.scheduling import run_online_schedule
from tests.test_ers import make_dag, make_runtime
from tests.test_serial_runtime import make_runtime as make_parallel_runtime, request


class TestERSRuntime(unittest.TestCase):
    def test_same_time_policy_calls_observe_same_state_before_any_action_is_submitted(self):
        runtime, members, grounds, bs = make_parallel_runtime(cores=2)
        dags = [request(m, g, inputs=(0., 0.), cycles=(10., 10.))
                for m, g in zip(members[:2], grounds[:2])]
        runtime.register_dags(dags)
        observations = []

        def choose(key):
            observations.append((key.owner_member_id, runtime.now, dict(runtime.active_tasks)))
            return PlacementDecision(bs)

        run_online_schedule(runtime, {d.owner_member: d.task_keys for d in dags}, choose)
        self.assertEqual(observations, [(0, 0., {}), (1, 0., {}), (0, 1., {}), (1, 1., {})])

    def test_interleaved_dags_are_dispatched_only_after_previous_task_finishes(self):
        traces = []
        for reverse in (False, True):
            runtime, ground, member, _ = make_runtime()
            requests = [DAGRequest(0, make_dag((4e8, 1e8)), member, ground),
                        DAGRequest(1, make_dag((2e8, 2e8)), member, ground)]
            if reverse:
                requests.reverse()
            plan = ERS(runtime).plan(requests)
            runtime.register_dags(requests)
            decisions = []

            def choose(key):
                decisions.append((key, runtime.now))
                return PlacementDecision(member)

            run_online_schedule(runtime, plan.member_orders, choose)
            result = runtime.finish_slot()
            self.assertEqual(len(result.finished_dags), 2)
            self.assertEqual(tuple(key for key, _ in decisions), plan.member_orders[member])
            for previous, (key, at) in zip(plan.order, decisions[1:]):
                self.assertAlmostEqual(at, runtime.trace(previous).compute_finish_at)
                self.assertAlmostEqual(runtime.trace(key).input_record.start_at, at)
            traces.append(decisions)
        self.assertEqual(*traces)

    def test_method_selects_ranked_task_before_local_compute(self):
        runtime, ground, member, _ = make_runtime()
        requests = [DAGRequest(0, make_dag((1e7,)), member, ground),
                    DAGRequest(1, make_dag((3e7,)), member, ground)]
        plan = ERS(runtime).plan(requests)
        runtime.register_dags(requests)
        run_online_schedule(runtime, plan.member_orders, lambda key: PlacementDecision(ground))
        self.assertEqual(runtime.trace(requests[1].task_keys[0]).compute_start_at, 0.)
        self.assertAlmostEqual(runtime.trace(requests[0].task_keys[0]).compute_start_at, .03)

    def test_input_and_dependency_share_two_hop_channels_after_decision(self):
        runtime, ground, member, bs = make_runtime(asymmetric=True)
        request = DAGRequest(0, make_dag(), member, ground)
        plan = ERS(runtime).plan([request])
        runtime.register_dags([request])
        run_online_schedule(runtime, plan.member_orders,
                            lambda key: PlacementDecision(ground if key.node_id == 0 else bs))
        parent, child = (runtime.trace(key) for key in request.task_keys)
        self.assertEqual(len(child.input_record.transfer_ids), 2)
        self.assertEqual(len(child.predecessor_records[parent.key].transfer_ids), 2)
        self.assertAlmostEqual(child.decision_at, .1)
        self.assertAlmostEqual(child.input_arrival_at, .16)
        self.assertAlmostEqual(child.predecessor_arrival_at[parent.key], .21)
        self.assertAlmostEqual(child.compute_start_at, .21)
        self.assertAlmostEqual(plan.costs[request.key].path_seconds_per_bit[ground, bs] * 1e6, .06)

    def test_zero_payload_keeps_dependencies_without_transfer_jobs(self):
        runtime, ground, member, bs = make_runtime()
        request = DAGRequest(0, make_dag(input_kbit=0., edge_kbit=0.), member, ground)
        runtime.register_dags([request])
        run_online_schedule(runtime, {member: request.task_keys},
                            lambda key: PlacementDecision(ground if key.node_id == 0 else bs))
        self.assertFalse(runtime.channels)
        self.assertAlmostEqual(runtime.trace(request.task_keys[1]).compute_start_at, .1)
