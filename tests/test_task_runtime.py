"""子任务数据到达状态测试。"""

import unittest

from env.runtime.task_runtime import TaskRuntime, TaskStatus
from env.types import EntityKind, EntityRef, TaskKey


class TestTaskRuntime(unittest.TestCase):
    def test_requires_input_and_all_predecessor_results_before_data_ready(self):
        task = TaskRuntime(
            key=TaskKey(0, 1, 2, 3),
            execution_node=EntityRef(EntityKind.BS, 0),
            ers_seq=5,
            cpu_cycles=10.0,
            input_bits=8.0,
            predecessors={TaskKey(0, 1, 2, 0), TaskKey(0, 1, 2, 1)},
        )
        parent_a, parent_b = sorted(task.predecessors)

        task.mark_input_arrived(4.0)
        task.mark_predecessor_arrived(parent_a, 3.0)
        self.assertIsNone(task.data_ready_at)
        task.mark_predecessor_arrived(parent_b, 5.0)

        self.assertEqual(task.data_ready_at, 5.0)
        self.assertEqual(task.status, TaskStatus.QUEUED_COMPUTE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
