"""共享频谱、发送端功率和异步传输的可手算回归场景。"""

from dataclasses import replace
import math
import unittest

import numpy as np

from env.communication.channel_model import ChannelModel
from env.contracts import PlacementDecision
from env.runtime.event_runtime import SchedulingRuntime, ServerSpec
from env.runtime.channel_queue import TransferStatus
from env.settings import ChannelConfig
from env.types import DirectedChannelKey, EntityKind, EntityRef
from tests.test_serial_runtime import make_runtime, request


class TestSharedCommunication(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c, self.d = (
            EntityRef(EntityKind.MEMBER_UAV, i) for i in range(4))
        self.bs = EntityRef(EntityKind.BS, 0)
        positions = dict(zip((self.a, self.b, self.c, self.d, self.bs),
                             map(np.array, ((0., 0., 50.), (1., 0., 50.),
                                            (0., 1., 50.), (-1., 0., 50.), (2., 0., 0.)))))
        config = replace(ChannelConfig.default(), total_bandwidth_mhz=12.,
                         reference_channel_gain=4e-11, noise_density_w_per_hz=1e-17)
        self.runtime = SchedulingRuntime(
            ChannelModel(config), positions,
            {node: ServerSpec((1.,), 0.) for node in positions},
            {node: 3. for node in positions}, max_duration_s=10.)

    def send(self, source, targets, bits):
        # 精确控制到达时刻，复用真实路由提交及事件执行路径。
        dag = request(source, source, dag_id=len(self.runtime.tasks), cycles=(100.,))
        key = self.runtime.register_dags([dag])[0]
        self.runtime._ready_members.clear()
        task = self.runtime.trace(key)
        task.execution_node = targets[-1]
        task.input_arrival_at = None
        route = tuple(DirectedChannelKey(u, v) for u, v in zip((source, *targets), targets))
        record = self.runtime._append_route(route, task, bits, is_input=True)
        self.runtime._dispatch(self.runtime.now)
        return record, [self.runtime.transfer_jobs[tid] for tid in record.transfer_ids]

    def test_asynchronous_example_preserves_progress_and_energy(self):
        runtime = self.runtime
        rb, (b,) = self.send(self.a, (self.b,), 24e6)
        runtime.advance_until(1.)
        rc, (c,) = self.send(self.a, (self.c,), 6e6)
        runtime.advance_until(1.5)
        rd, (d,) = self.send(self.a, (self.d,), 6e6)
        self.assertAlmostEqual(b.remaining_bits, 9e6)
        self.assertAlmostEqual(c.remaining_bits, 3e6)
        self.assertEqual(sum(j.bandwidth_hz for j in (b, c, d)), 12e6)
        self.assertEqual(sum(j.power_w for j in (b, c, d)), 3.)
        runtime.advance_until(2.)  # 原 B 完成事件已经失效。
        self.assertIsNone(rb.finish_at)
        runtime.advance_until(3.)
        self.assertEqual([rc.finish_at, rd.finish_at, rb.finish_at], [2.25, 2.75, 3.])
        self.assertEqual([j.remaining_bits for j in (b, c, d)], [0., 0., 0.])
        self.assertAlmostEqual(sum(r.tx_energy_j for r in (rb, rc, rd)), 9.)
        self.assertAlmostEqual(runtime.finish_slot().tx_energy_j, 9.)

    def test_coincident_member_transfer_finishes_and_releases_bandwidth(self):
        self.runtime.update_entity_positions({self.b: self.runtime.entity_positions[self.a]})
        fast_record, (fast,) = self.send(self.a, (self.b,), 500_000.)
        _, (slow,) = self.send(self.a, (self.c,), 24e6)
        finish = fast.predicted_finish_at
        self.assertTrue(math.isfinite(finish))
        self.assertGreater(finish, self.runtime.now)
        self.runtime.advance_until(finish)
        self.assertEqual(fast.status, TransferStatus.FINISHED)
        self.assertEqual(fast_record.finish_at, finish)
        self.assertEqual(fast.remaining_bits, 0.)
        self.assertEqual(slow.bandwidth_hz, 12e6)
        self.assertEqual(slow.power_w, 3.)
        self.assertAlmostEqual(fast.tx_energy_j, 1.5 * finish)

    def test_global_bandwidth_but_power_belongs_to_each_sender(self):
        _, (ab,) = self.send(self.a, (self.b,), 24e6)
        _, (ac,) = self.send(self.a, (self.c,), 24e6)
        _, (db,) = self.send(self.d, (self.b,), 24e6)
        self.assertEqual([j.bandwidth_hz for j in (ab, ac, db)], [4e6] * 3)
        self.assertEqual([j.power_w for j in (ab, ac, db)], [1.5, 1.5, 3.])
        self.assertAlmostEqual(ab.rate_bps, 4e6 * math.log2(2.5))

    def test_queue_and_unready_relay_do_not_reserve_resources(self):
        first, (ab, bc) = self.send(self.a, (self.b, self.c), 12e6)
        second, (queued,) = self.send(self.a, (self.b,), 12e6)
        self.assertEqual(ab.bandwidth_hz, 12e6)
        self.assertEqual(bc.bandwidth_hz, 0.)
        self.assertEqual(queued.bandwidth_hz, 0.)
        self.assertIsNone(bc.source_ready_at)
        self.runtime.advance_until(1.)
        self.assertEqual(ab.status, TransferStatus.FINISHED)
        self.assertEqual(bc.start_at, 1.)
        self.assertEqual(queued.start_at, 1.)
        self.assertEqual(bc.bandwidth_hz, 6e6)
        self.assertEqual(queued.bandwidth_hz, 6e6)
        self.runtime.finish_slot()
        self.assertIsNotNone(first.finish_at)
        self.assertIsNotNone(second.finish_at)

    def test_cutoff_settles_partial_progress_once(self):
        self.runtime._deadline = 1.5
        record, (job,) = self.send(self.a, (self.b,), 24e6)
        self.runtime.advance_until(1.)
        self.assertEqual(record.tx_energy_j, 3.)
        result = self.runtime.finish_slot()
        self.assertEqual(job.remaining_bits, 6e6)
        self.assertEqual(job.tx_energy_j, 4.5)
        self.assertIsNone(job.finish_at)
        self.assertIsNone(record.finish_at)
        self.assertEqual(result.tx_energy_j, 4.5)
        self.assertIs(result, self.runtime.finish_slot())

    def test_simultaneous_finish_at_deadline_counts_both(self):
        self.runtime._deadline = 2.
        first, _ = self.send(self.a, (self.b,), 12e6)
        second, _ = self.send(self.a, (self.c,), 12e6)
        result = self.runtime.finish_slot()
        self.assertEqual(first.finish_at, 2.)
        self.assertEqual(second.finish_at, 2.)
        self.assertEqual(result.tx_energy_j, 6.)

    def test_batch_order_does_not_change_rates_or_decision_time(self):
        outcomes = []
        for reverse in (False, True):
            runtime, members, grounds, bs = make_runtime()
            dags = [request(m, g, inputs=(1., 0.), cycles=(10., 10.))
                    for m, g in zip(members[:2], grounds[:2])]
            runtime.register_dags(dags)
            placements = {d.task_keys[0]: PlacementDecision(d.owner_member)
                          for d in (reversed(dags) if reverse else dags)}
            runtime.submit_placements(placements)
            self.assertEqual(runtime.advance_to_decision(), (0, 1))
            # 两个发送端各用完整功率，带宽减半：上传耗时 2/log2(3)。
            self.assertAlmostEqual(runtime.now, 1. + 2. / math.log2(3.))
            outcomes.append([(runtime.trace(d.task_keys[0]).input_arrival_at,
                              runtime.trace(d.task_keys[0]).compute_finish_at) for d in dags])
        self.assertEqual(*outcomes)

    def test_observation_time_splits_do_not_double_count_progress_or_energy(self):
        first, (ab,) = self.send(self.a, (self.b,), 24e6)
        for timestamp in (.1, .3, .6, 1.):
            self.runtime.advance_until(timestamp)
        self.assertAlmostEqual(ab.remaining_bits, 12e6)
        self.assertAlmostEqual(first.tx_energy_j, 3.)
        second, _ = self.send(self.a, (self.c,), 6e6)
        for timestamp in (1.2, 1.5, 1.9, 2., 2.5):
            self.runtime.advance_until(timestamp)
        self.assertAlmostEqual(second.finish_at, 2.)
        self.assertAlmostEqual(first.finish_at, 2.5)
        self.assertAlmostEqual(self.runtime.finish_slot().tx_energy_j, 7.5)
