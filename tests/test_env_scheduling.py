"""环境提交接口的异步决策与截止行为。"""

import unittest

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.runtime.task_runtime import TaskStatus
from env.settings import EnvConfig
from tests.test_serial_runtime import make_runtime, request


class TestEnvScheduling(unittest.TestCase):
    def make_environment(self, *, duration=30., cores=1):
        runtime, members, grounds, bs = make_runtime(duration=duration, cores=cores)
        environment = MRDTSEnv({"env": EnvConfig.default()})
        environment._runtime = runtime
        return environment, runtime, members, grounds, bs

    def test_submit_returns_next_owner_without_waiting_for_whole_batch(self):
        environment, runtime, members, grounds, bs = self.make_environment()
        slow = request(members[0], grounds[0], inputs=(0., 0.), cycles=(50., 10.))
        fast = request(members[1], grounds[1], inputs=(2., 2.), cycles=(10., 10.))
        runtime.register_dags([slow, fast])

        ready = environment.submit_schedule({
            slow.task_keys[0]: PlacementDecision(bs),
            fast.task_keys[0]: PlacementDecision(grounds[1]),
        })

        self.assertEqual(ready, (1,))
        self.assertEqual(runtime.now, 1.)
        self.assertEqual(runtime.active_tasks, {0: slow.task_keys[0]})
        self.assertEqual(runtime.trace(fast.task_keys[0]).input_record.transfer_ids, ())
        ready = environment.submit_schedule({fast.task_keys[1]: PlacementDecision(grounds[1])})
        self.assertEqual(ready, (0,))
        self.assertEqual(runtime.now, 5.)
        self.assertEqual(runtime.trace(fast.task_keys[1]).compute_finish_at, 2.)
        ready = environment.submit_schedule({slow.task_keys[1]: PlacementDecision(members[0])})
        self.assertEqual(ready, ())
        self.assertEqual(runtime.now, 6.)
        self.assertFalse(runtime.closed)
        self.assertEqual(environment.step_index, 0)
        self.assertEqual(environment._next_slot_start, 0.)
        self.assertEqual(len(runtime.finish_slot().finished_dags), 2)

    def test_simultaneous_and_zero_duration_completions_settle_before_return(self):
        environment, runtime, members, grounds, bs = self.make_environment(cores=2)
        dags = [request(m, g, inputs=(0., 0., 0.), cycles=(10., 0., 10.),
                        edges=((0, 1), (1, 2)), sizes=(0., 0.))
                for m, g in zip(members[:2], grounds[:2])]
        runtime.register_dags(dags)

        for index in (0, 1):
            ready = environment.submit_schedule({
                dag.task_keys[index]: PlacementDecision(bs) for dag in reversed(dags)
            })
            self.assertEqual(ready, (0, 1))
            self.assertEqual(runtime.now, 1.)
            self.assertFalse(runtime.active_tasks)
            self.assertFalse(runtime.servers[bs].running)
            self.assertEqual(runtime.servers[bs].compute_wait(runtime.now), 0.)
            for dag in dags:
                self.assertIs(runtime.trace(dag.task_keys[index]).status, TaskStatus.FINISHED)
                self.assertIsNone(runtime.trace(dag.task_keys[index + 1]).decision_at)

    def test_deadline_returns_no_decisions_and_keeps_settlement_available(self):
        for cycles, finished in ((10., True), (20., False)):
            with self.subTest(cycles=cycles):
                environment, runtime, members, grounds, bs = self.make_environment(duration=1.)
                dag = request(members[0], grounds[0], inputs=(0., 0.), cycles=(cycles, 10.))
                runtime.register_dags([dag])

                ready = environment.submit_schedule({dag.task_keys[0]: PlacementDecision(bs)})

                self.assertEqual(ready, ())
                self.assertEqual(runtime.now, 1.)
                self.assertTrue(runtime.closed)
                self.assertIs(runtime.trace(dag.task_keys[0]).status,
                              TaskStatus.FINISHED if finished else TaskStatus.FAILED)
                self.assertIs(runtime.trace(dag.task_keys[1]).status, TaskStatus.FAILED)
                self.assertIsNone(runtime.trace(dag.task_keys[1]).decision_at)
                result = runtime.finish_slot()
                self.assertEqual(result.failed_dags, (dag.key,))
                self.assertEqual(result.compute_energy_j, 0.)  # BS 仍计算，但不统计能耗。
                self.assertIs(result, runtime.finish_slot())
                self.assertEqual(environment.step_index, 0)


if __name__ == "__main__":
    unittest.main()
