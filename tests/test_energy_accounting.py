"""验证按事件入账、逐跳归属及截止部分结算。"""

import unittest

import numpy as np

from env.contracts import PlacementDecision
from tests.test_serial_runtime import make_runtime, request


class TestEnergyAccounting(unittest.TestCase):
    def make_runtime(self, duration=10.):
        self.queue = np.zeros(4)
        self.events = []

        def energy(node, amount):
            self.queue[node.index] += amount
            self.events.append((runtime.now, node.index, amount))

        def elapsed(seconds):
            self.queue[:] += 2. * seconds

        runtime, members, grounds, bs = make_runtime(
            duration=duration, on_member_energy=energy, on_time_advance=elapsed,
        )
        return runtime, members, grounds, bs

    def test_compute_charged_on_completion_to_executor_before_next_decision(self):
        runtime, members, grounds, _ = self.make_runtime()
        dag = request(members[0], grounds[0], inputs=(0., 0.), cycles=(20., 10.))
        parent, child = runtime.register_dags([dag])
        runtime.submit_placements({parent: PlacementDecision(members[1])})
        self.assertEqual(runtime.trace(parent).compute_energy_j, 0.)
        np.testing.assert_array_equal(self.queue, 0.)
        runtime.advance_until(1.)
        np.testing.assert_array_equal(self.queue, 2.)
        self.assertEqual(runtime.advance_to_decision(), (0,))
        np.testing.assert_array_equal(self.queue, [4., 24., 4., 4.])
        self.assertEqual(self.events, [(2., 1, 20.)])
        self.assertEqual(runtime.trace(parent).compute_energy_j, 20.)
        runtime.submit_placements({child: PlacementDecision(grounds[0])})
        runtime.advance_to_decision()
        result = runtime.finish_slot()
        self.assertEqual(result.compute_energy_j, 20.)
        np.testing.assert_array_equal(self.queue, [20., 40., 20., 20.])
        runtime.finish_slot()
        self.assertEqual(len(self.events), 1)
        row = next(row for row in result.member_energy if row.member_id == 1)
        self.assertEqual(row.compute_energy_j, 20.)

    def test_each_relay_hop_charged_on_finish_and_ground_bs_are_excluded(self):
        runtime, members, grounds, bs = self.make_runtime(duration=16.)
        dag = request(members[0], grounds[0], inputs=(0., 0.), cycles=(10., 10.),
                      edges=((0, 1),), sizes=(2.,))
        parent, child = runtime.register_dags([dag])
        runtime.submit_placements({parent: PlacementDecision(members[1])})
        runtime.advance_to_decision()
        runtime.submit_placements({child: PlacementDecision(grounds[0])})
        record = runtime.trace(child).predecessor_records[parent]
        runtime.advance_until(2.)
        self.assertEqual(record.tx_energy_j, 0.)
        runtime.advance_until(3.)  # Member 1 -> Member 0 完成，用户还没收到数据。
        self.assertEqual(record.tx_energy_j, 2.)
        self.assertIsNone(record.finish_at)
        self.assertEqual(self.events[-1], (3., 1, 2.))
        runtime.advance_until(5.)
        self.assertEqual(record.tx_energy_j, 4.)
        self.assertEqual(self.events[-1], (5., 0, 2.))
        runtime.advance_to_decision()
        other = request(members[0], grounds[0], dag_id=1, inputs=(2., 0.), cycles=(10., 10.),
                        edges=((0, 1),), sizes=(1.,))
        key = runtime.register_dags([other])[0]
        runtime.submit_placements({key: PlacementDecision(bs)})
        runtime.advance_to_decision()
        runtime.submit_placements({other.task_keys[1]: PlacementDecision(grounds[0])})
        runtime.advance_to_decision()
        result = runtime.finish_slot()
        self.assertEqual(result.compute_energy_j, 10.)
        self.assertEqual(runtime.trace(key).compute_energy_j, 0.)
        self.assertAlmostEqual(runtime.trace(key).input_record.tx_energy_j, 2.)
        self.assertAlmostEqual(result.tx_energy_j, 7.)  # BS 下发第一跳不计，Member 转发计 1J。

    def test_deadline_counts_partial_work_once_and_ignores_unstarted_work(self):
        runtime, members, grounds, bs = self.make_runtime(duration=2.)
        dags = [
            request(members[0], grounds[0], inputs=(1.,), cycles=(10.,)),
            request(members[1], grounds[1], cycles=(50.,)),
            request(members[2], grounds[2], cycles=(10.,)),
            request(members[3], grounds[3], cycles=(20.,)),
        ]
        runtime.register_dags(dags)
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(target)
                                   for dag, target in zip(dags, (bs, members[1], members[1], members[3]))})
        runtime.advance_to_decision()
        result = runtime.finish_slot()
        # 用户上传 1s 不计能耗；Member 0 转发恰好在截止完成，记 1J。
        self.assertAlmostEqual(result.tx_energy_j, 1.)
        # Member 1 执行 2/5，Member 3 恰好完成；等待核心的任务为 0。
        self.assertAlmostEqual(result.compute_energy_j, 40.)
        np.testing.assert_allclose(self.queue, [5., 24., 4., 24.])
        snapshot = self.queue.copy()
        self.assertIs(result, runtime.finish_slot())
        np.testing.assert_array_equal(self.queue, snapshot)

    def test_deadline_charges_only_elapsed_part_of_member_transmission(self):
        runtime, members, grounds, bs = self.make_runtime(duration=3.)
        dag = request(members[0], grounds[0], inputs=(2.,), cycles=(10.,))
        key = runtime.register_dags([dag])[0]
        runtime.submit_placements({key: PlacementDecision(bs)})
        runtime.advance_to_decision()
        result = runtime.finish_slot()
        self.assertAlmostEqual(result.tx_energy_j, 1.)
        self.assertEqual(result.compute_energy_j, 0.)
        np.testing.assert_allclose(self.queue, [7., 6., 6., 6.])


if __name__ == '__main__':
    unittest.main()
