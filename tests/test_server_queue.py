"""多核服务器 FIFO 队列测试。"""

import unittest

from env.runtime.server_queue import ServerState
from env.types import EntityKind, EntityRef, TaskKey


class TestServerState(unittest.TestCase):
    def test_two_cores_take_first_two_fifo_entries(self):
        server = ServerState(EntityRef(EntityKind.BS, 0), (10.0, 10.0), 0.1)
        tasks = [TaskKey(0, 0, 0, index) for index in range(3)]
        for index, task in enumerate(tasks):
            server.enqueue(task, cpu_cycles=10.0, ready_at=0.0)

        started = server.dispatch(0.0)

        self.assertEqual([record.task_key for record in started], tasks[:2])
        self.assertEqual(server.queued_task_keys, (tasks[2],))

    def test_later_arrival_never_bypasses_existing_fifo_head(self):
        server = ServerState(EntityRef(EntityKind.MEMBER_UAV, 0), (10.0,), 0.1)
        first = TaskKey(0, 0, 0, 0)
        second = TaskKey(0, 0, 0, 1)
        server.enqueue(first, cpu_cycles=10.0, ready_at=0.0)
        running = server.dispatch(0.0)[0]
        server.enqueue(second, cpu_cycles=10.0, ready_at=0.1)

        server.finish(running.core_id, 1.0)
        started = server.dispatch(1.0)

        self.assertEqual(started[0].task_key, second)
        self.assertEqual(started[0].start_at, 1.0)

    def test_compute_wait_predicts_shared_multicore_queue_without_mutation(self):
        server = ServerState(EntityRef(EntityKind.MEMBER_UAV, 0), (10.0, 10.0), 0.1)
        tasks = [TaskKey(0, 0, 0, index) for index in range(5)]
        for task, cycles in zip(tasks, (20., 50.)):
            server.enqueue(task, cycles, 10.)
        server.dispatch(10.)
        for task, cycles in zip(tasks[2:], (40., 10.)):
            server.enqueue(task, cycles, 10.)
        queued = server.queued_task_keys
        running = server.running.copy()

        self.assertEqual(server.compute_wait(10.), 6.)
        self.assertEqual(server.compute_wait(11.), 5.)
        self.assertEqual(server.queued_task_keys, queued)
        self.assertEqual(server.running, running)

        # 将假想任务实际入队，用真实派发结果核对预测。
        server.enqueue(tasks[4], 10., 10.)
        for at, cores in ((12., (0,)), (15., (1,)), (16., (0, 1))):
            for core_id in cores:
                server.finish(core_id, at)
            started = server.dispatch(at)
        self.assertEqual(started[0].task_key, tasks[4])
        self.assertEqual(started[0].start_at, 16.)

    def test_compute_wait_is_zero_with_an_idle_core(self):
        server = ServerState(EntityRef(EntityKind.MEMBER_UAV, 0), (10., 10.), 0.1)
        self.assertEqual(server.compute_wait(10.), 0.)
        server.enqueue(TaskKey(0, 0, 0, 0), 100., 10.)
        server.dispatch(10.)
        self.assertEqual(server.compute_wait(11.), 0.)

    def test_compute_wait_respects_core_frequencies_and_simultaneous_order(self):
        server = ServerState(EntityRef(EntityKind.BS, 0), (10., 20.), 0.1)
        tasks = [TaskKey(0, 0, 0, index) for index in range(4)]
        for task, cycles in zip(tasks, (20., 40.)):
            server.enqueue(task, cycles, 10.)
        server.dispatch(10.)
        for task, cycles in zip(tasks[2:], (100., 20.)):
            server.enqueue(task, cycles, 10.)
        self.assertEqual(server.compute_wait(10.), 3.)

    def test_compute_wait_settles_zero_duration_after_dispatching_idle_cores(self):
        server = ServerState(EntityRef(EntityKind.BS, 0), (10., 1.), 0.1)
        tasks = [TaskKey(0, 0, 0, index) for index in range(4)]
        for task, cycles in zip(tasks, (0., 10., 100.)):
            server.enqueue(task, cycles, 10.)
        self.assertEqual(server.compute_wait(10.), 10.)
        server.enqueue(tasks[3], 10., 10.)
        server.dispatch(10.)
        server.finish(0, 10.)
        server.dispatch(10.)
        for core_id in (0, 1):
            server.finish(core_id, 20.)
        self.assertEqual(server.dispatch(20.)[0].task_key, tasks[3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
