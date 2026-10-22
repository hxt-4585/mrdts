"""三个可手算的 ERS 在线卸载示例；时间单位为秒。

复用测试场景：每跳速率 1 kbit/s，每个核心 10 cycle/s。
固定策略只选择执行节点；排序与事件推进使用真实 ERS 和 runtime。
运行：python -m unittest tests.test_online_scheduling_examples -v
"""

import unittest

from env.contracts import PlacementDecision
from methods.random.ers import ERS
from methods.scheduling import run_online_schedule
from tests.test_serial_runtime import make_runtime, request


class TestOnlineSchedulingExamples(unittest.TestCase):
    def run_scenario(self, runtime, requests, targets):
        plan = ERS(runtime).plan(requests)
        runtime.register_dags(requests)
        decisions = {member: [] for member in plan.member_orders}

        def choose(key):
            member = next(m for m in decisions if m.index == key.owner_member_id)
            previous = decisions[member]
            self.assertNotIn(member.index, runtime.active_tasks)
            if previous:
                self.assertAlmostEqual(runtime.trace(previous[-1]).compute_finish_at, runtime.now)
            previous.append(key)
            return PlacementDecision(targets[key])

        run_online_schedule(runtime, plan.member_orders, choose)
        for member, order in plan.member_orders.items():
            self.assertEqual(tuple(decisions[member]), order)
        result = runtime.finish_slot()
        self.assertEqual(len(result.finished_dags), len(requests))
        self.assertFalse(result.failed_dags)
        return plan

    def assert_task_times(self, task, decision, ready, start, finish):
        for field, expected in zip(
            ('decision_at', 'data_ready_at', 'compute_start_at', 'compute_finish_at'),
            (decision, ready, start, finish),
        ):
            with self.subTest(task=task.key, field=field):
                self.assertAlmostEqual(getattr(task, field), expected)

    def assert_transfer_times(self, runtime, record, expected):
        self.assertEqual(len(record.transfer_ids), len(expected))
        for transfer_id, (start, finish) in zip(record.transfer_ids, expected):
            job = runtime.transfer_jobs[transfer_id]
            self.assertAlmostEqual(job.start_at, start)
            self.assertAlmostEqual(job.finish_at, finish)

    def test_1_later_decision_runs_first_with_one_or_two_cores(self):
        """m 在 1 决策、4 到达；n 在 2 决策并就绪，先获得 B 的核心。"""
        for cores in (1, 2):
            with self.subTest(cores=cores):
                runtime, members, grounds, _ = make_runtime(cores=cores)
                a, b = members[:2]
                dag_a = request(a, grounds[0], inputs=(0., 1.5), cycles=(10., 10.),
                                edges=((0, 1),), sizes=(0.,))
                dag_b = request(b, grounds[1], inputs=(0., 0.), cycles=(20., 30.),
                                edges=((0, 1),), sizes=(0.,))
                prep_a, m = dag_a.task_keys
                prep_b, n = dag_b.task_keys
                plan = self.run_scenario(runtime, [dag_a, dag_b],
                                         {prep_a: a, m: b, prep_b: b, n: b})
                self.assertEqual(plan.member_orders[a], (prep_a, m))
                self.assertEqual(plan.member_orders[b], (prep_b, n))
                self.assert_task_times(runtime.trace(prep_a), 0., 0., 0., 1.)
                self.assert_task_times(runtime.trace(prep_b), 0., 0., 0., 2.)
                self.assert_task_times(runtime.trace(n), 2., 2., 2., 5.)
                start = 5. if cores == 1 else 4.
                self.assert_task_times(runtime.trace(m), 1., 4., start, start + 1.)
                self.assertEqual(runtime.trace(m).execution_core_id, cores - 1)
                self.assertEqual(runtime.trace(n).execution_core_id, 0)
                self.assert_transfer_times(runtime, runtime.trace(m).input_record,
                                           ((1., 2.5), (2.5, 4.)))

    def test_2_input_and_result_overlap_then_share_the_relay_channel(self):
        """输入 G→A 与结果 A→B 同时启动；输入的第二跳等待结果传完。"""
        runtime, members, grounds, _ = make_runtime(cores=2)
        a, b = members[:2]
        dag = request(a, grounds[0], inputs=(0., 2.), cycles=(10., 10.),
                      edges=((0, 1),), sizes=(3.,))
        parent, child = dag.task_keys
        plan = self.run_scenario(runtime, [dag], {parent: a, child: b})
        self.assertEqual(plan.member_orders[a], (parent, child))
        self.assert_task_times(runtime.trace(parent), 0., 0., 0., 1.)
        task = runtime.trace(child)
        self.assert_transfer_times(runtime, task.input_record, ((1., 3.), (4., 6.)))
        self.assert_transfer_times(runtime, task.predecessor_records[parent], ((1., 4.),))
        second_hop = runtime.transfer_jobs[task.input_record.transfer_ids[1]]
        self.assertAlmostEqual(second_hop.source_ready_at, 3.)
        self.assertAlmostEqual(task.input_arrival_at, 6.)
        self.assertAlmostEqual(task.predecessor_arrival_at[parent], 4.)
        self.assert_task_times(task, 1., 6., 6., 7.)

    def test_3_same_channel_serializes_while_another_destination_runs_in_parallel(self):
        """3 时刻：A→B 依次传两份结果，A→C 同时传另一份结果。"""
        runtime, members, grounds, _ = make_runtime(cores=2)
        a, b, c, d = members
        dag_a = request(a, grounds[0], inputs=(0., 0., 0.), cycles=(20., 10., 10.),
                        edges=((0, 1), (0, 2), (1, 2)), sizes=(0., 2., 3.))
        dag_d = request(d, grounds[3], inputs=(0., 0.), cycles=(30., 10.),
                        edges=((0, 1),), sizes=(4.,))
        p1, p2, target_b = dag_a.task_keys
        q, target_c = dag_d.task_keys
        plan = self.run_scenario(runtime, [dag_a, dag_d],
                                 {p1: a, p2: a, target_b: b, q: a, target_c: c})
        self.assertEqual(plan.member_orders[a], (p1, p2, target_b))
        self.assertEqual(plan.member_orders[d], (q, target_c))
        self.assert_task_times(runtime.trace(p1), 0., 0., 0., 2.)
        self.assert_task_times(runtime.trace(p2), 2., 2., 2., 3.)
        self.assert_task_times(runtime.trace(q), 0., 0., 0., 3.)
        task_b, task_c = runtime.trace(target_b), runtime.trace(target_c)
        self.assert_transfer_times(runtime, task_b.predecessor_records[p1], ((3., 5.),))
        self.assert_transfer_times(runtime, task_b.predecessor_records[p2], ((5., 8.),))
        self.assert_transfer_times(runtime, task_c.predecessor_records[q], ((3., 7.),))
        self.assert_task_times(task_b, 3., 8., 8., 9.)
        self.assert_task_times(task_c, 3., 7., 7., 8.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
