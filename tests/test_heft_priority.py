"""HEFT upward-rank 优先级测试。"""

import unittest

import numpy as np

from env.dag_generator import DAG
from methods.heft_priority import HEFTPriority


class TestHEFTPriority(unittest.TestCase):
    def test_parent_precedes_its_successor(self):
        dag = DAG(
            node_num=3,
            edges=[(0, 1), (1, 2)],
            node_features=np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]),
            edge_features={(0, 1): 1.0, (1, 2): 1.0},
        )

        ordered = HEFTPriority.order(dag, average_compute_s={0: 2.0, 1: 3.0, 2: 4.0}, average_edge_comm_s={(0, 1): 1.0, (1, 2): 1.0})

        self.assertEqual(ordered, (0, 1, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
