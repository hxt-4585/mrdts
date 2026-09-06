"""DAG 生成逻辑的单元测试。

用法：
    python tests/test_dag_generation.py
"""

import os
import sys
import unittest
from dataclasses import replace

import numpy as np

# 将项目根目录加入 sys.path，便于导入 env 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.randomness import RandomStreams
from env.settings import DAGConfig
from env.workload.dag_generator import DAGGenerator


class TestDAGGeneration(unittest.TestCase):
    """DAG 生成器的正确性测试。"""

    def setUp(self):
        self.cfg = DAGConfig.default()
        self.gen = DAGGenerator(self.cfg, rng=RandomStreams.from_config().dag, py_rng=RandomStreams.from_config().dag_python)

    def test_node_num(self):
        """每个 DAG 的节点数应等于配置的 n。"""
        for _ in range(20):
            dag = self.gen.generate_single_dag()
            self.assertEqual(dag.node_num, self.cfg.n)

    def test_acyclic(self):
        """分层生成保证所有边 u < v，因此必为无环图。"""
        for _ in range(20):
            dag = self.gen.generate_single_dag()
            for u, v in dag.edges:
                self.assertLess(u, v, f"存在反向边或环: ({u}, {v})")

    def test_out_degree_bound(self):
        """每个节点的出度不应超过 max_out。"""
        for _ in range(20):
            dag = self.gen.generate_single_dag()
            out_degree = {}
            for u, _ in dag.edges:
                out_degree[u] = out_degree.get(u, 0) + 1
            for deg in out_degree.values():
                self.assertLessEqual(deg, self.cfg.max_out)

    def test_node_features_shape_and_range(self):
        """节点特征矩阵 shape 与取值范围应正确。"""
        for _ in range(20):
            dag = self.gen.generate_single_dag()
            self.assertEqual(dag.node_features.shape, (self.cfg.n, 2))
            # 第 0 列：输入数据量 (KB)
            self.assertTrue(
                (dag.node_features[:, 0] >= self.cfg.input_data_range[0]).all()
                and (dag.node_features[:, 0] <= self.cfg.input_data_range[1]).all()
            )
            # 第 1 列：计算量 (CPU cycles)
            self.assertTrue(
                (dag.node_features[:, 1] >= self.cfg.cpu_range[0]).all()
                and (dag.node_features[:, 1] <= self.cfg.cpu_range[1]).all()
            )

    def test_edge_features_range(self):
        """边特征（中间结果数据量）取值范围应正确。"""
        for _ in range(20):
            dag = self.gen.generate_single_dag()
            low, high = self.cfg.intermediate_data_range
            for u, v in dag.edges:
                val = dag.edge_features[(u, v)]
                self.assertGreaterEqual(val, low)
                self.assertLessEqual(val, high)

    def test_generate_batch(self):
        """批量生成的数量与每个 DAG 的节点数应正确。"""
        dag_list = self.gen.generate(30)
        self.assertEqual(len(dag_list), 30)
        for dag in dag_list:
            self.assertEqual(dag.node_num, self.cfg.n)

    def test_extreme_rho_no_hang(self):
        """极端 rho 取值下不应死循环（旧实现存在此缺陷）。"""
        for rho in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            cfg = replace(self.cfg, rho=rho)
            gen = DAGGenerator(cfg, rng=RandomStreams.from_config().dag, py_rng=RandomStreams.from_config().dag_python)
            dag = gen.generate_single_dag()
            self.assertEqual(dag.node_num, cfg.n)
            for u, v in dag.edges:
                self.assertLess(u, v)

    def test_reproducible_with_seed(self):
        """固定随机种子后，生成结果应可复现。"""

        def gen_one():
            dag = DAGGenerator(DAGConfig.default(), rng=RandomStreams.from_config().dag, py_rng=RandomStreams.from_config().dag_python).generate_single_dag()
            return list(dag.edges), dag.node_features.copy(), dict(dag.edge_features)

        edges_a, feats_a, edge_feats_a = gen_one()
        edges_b, feats_b, edge_feats_b = gen_one()
        self.assertEqual(edges_a, edges_b)
        np.testing.assert_array_equal(feats_a, feats_b)
        self.assertEqual(edge_feats_a, edge_feats_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
