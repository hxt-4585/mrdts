"""多核服务器 FIFO 队列测试。"""

import unittest

from env.channel_queue import EntityKind, EntityRef
from env.server_queue import ServerState
from env.task_runtime import TaskKey


class TestServerState(unittest.TestCase):
    def test_two_cores_take_first_two_fifo_entries(self):
        server = ServerState(EntityRef(EntityKind.BS, 0), (10.0, 10.0), 0.1)
        tasks = [TaskKey(0, 0, 0, index) for index in range(3)]
        for index, task in enumerate(tasks):
            server.enqueue(task, cpu_cycles=10.0, ready_at=0.0, ers_seq=index)

        started = server.dispatch(0.0)

        self.assertEqual([record.task_key for record in started], tasks[:2])
        self.assertEqual(server.queued_task_keys, (tasks[2],))

    def test_later_arrival_never_bypasses_existing_fifo_head(self):
        server = ServerState(EntityRef(EntityKind.MEMBER_UAV, 0), (10.0,), 0.1)
        first = TaskKey(0, 0, 0, 0)
        second = TaskKey(0, 0, 0, 1)
        server.enqueue(first, cpu_cycles=10.0, ready_at=0.0, ers_seq=0)
        running = server.dispatch(0.0)[0]
        server.enqueue(second, cpu_cycles=10.0, ready_at=0.1, ers_seq=1)

        server.finish(running.core_id, 1.0)
        started = server.dispatch(1.0)

        self.assertEqual(started[0].task_key, second)
        self.assertEqual(started[0].start_at, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
