"""DAG 运行时依赖状态的单元测试。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dag_generator import DAG
from env.dag_runtime import DAGRuntime


def make_diamond_dag():
    """构造 0 -> {1, 2} -> 3 的最小汇合 DAG。"""
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    return DAG(
        node_num=4,
        edges=edges,
        node_features=np.zeros((4, 2)),
        edge_features={edge: 0 for edge in edges},
    )


class TestDAGRuntime(unittest.TestCase):
    def test_releases_successor_only_after_all_predecessors_finish(self):
        """汇合节点必须等待其全部前驱完成才变为就绪。"""
        runtime = DAGRuntime(make_diamond_dag())

        self.assertEqual(runtime.ready_node_ids, {0})

        runtime.mark_finished(0, 1.0)
        self.assertEqual(runtime.ready_node_ids, {1, 2})

        runtime.mark_finished(1, 3.0)
        self.assertEqual(runtime.ready_node_ids, {2})

        runtime.mark_finished(2, 2.0)
        self.assertEqual(runtime.ready_node_ids, {3})

    def test_reports_completion_time_after_every_node_finishes(self):
        """DAG 完成时刻应是所有子任务完成时刻的最大值。"""
        runtime = DAGRuntime(make_diamond_dag())

        runtime.mark_finished(0, 1.0)
        runtime.mark_finished(1, 3.0)
        runtime.mark_finished(2, 2.0)
        runtime.mark_finished(3, 4.5)

        self.assertTrue(runtime.is_finished)
        self.assertEqual(runtime.completion_time, 4.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
