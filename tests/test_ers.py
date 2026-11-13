"""逐核心、中继路径和多 DAG ERS 排序的可手算测试。"""

import unittest

import numpy as np

from env.communication.channel_model import ChannelModel
from env.contracts import DAGRequest
from env.runtime.event_runtime import SchedulingRuntime, ServerSpec
from env.types import DirectedChannelKey, EntityKind, EntityRef, TaskKey
from env.workload.dag_generator import DAG
from scripts.ers import ERS


def make_runtime(*, asymmetric=False):
    """使用全网同一带宽；可用不同设备功率构造非对称通信代价。"""
    ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
    member = EntityRef(EntityKind.MEMBER_UAV, 0)
    bs = EntityRef(EntityKind.BS, 0)
    positions = {ground: np.array([0., 0., 0.]), member: np.array([0., 0., 100.]),
                 bs: np.array([100., 0., 20.])}
    model = ChannelModel()
    powers = {ground: .5 if asymmetric else 1., member: 1., bs: 2. if asymmetric else 1.}
    runtime = SchedulingRuntime(
        model, positions,
        {ground: ServerSpec((1e9,), 0.), member: ServerSpec((10e9, 10e9), 1e-28),
         bs: ServerSpec((12e9,) * 4, 1e-28)},
        powers,
        member_regions={member: 1})
    return runtime, ground, member, bs


def hop_cost(runtime, source, target):
    return runtime.channel_model.transmission_delay_s(
        1., runtime.entity_positions[source], runtime.entity_positions[target],
        runtime.transmit_powers[source], SchedulingRuntime._link_type(DirectedChannelKey(source, target)))


def weighted_comm(runtime, ground, member, bs, bits):
    # 1/2/4 个核心的所有有序位置对：地面链路权重 6，BS 链路权重 12。
    ground_pair = hop_cost(runtime, ground, member) + hop_cost(runtime, member, ground)
    bs_pair = hop_cost(runtime, member, bs) + hop_cost(runtime, bs, member)
    return bits * (6 * ground_pair + 12 * bs_pair) / 49


def make_dag(cycles=(1e8, 1e8), input_kbit=1000.0, edge_kbit=1000.0):
    edges = [(i, i + 1) for i in range(len(cycles) - 1)]
    return DAG(len(cycles), edges, np.array([[input_kbit, c] for c in cycles]),
               {edge: edge_kbit for edge in edges})


class TestERS(unittest.TestCase):
    def test_parent_precedes_its_successor(self):
        dag = DAG(
            node_num=3,
            edges=[(0, 1), (1, 2)],
            node_features=np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]),
            edge_features={(0, 1): 1.0, (1, 2): 1.0},
        )

        ordered = ERS.order(dag, average_compute_s={0: 2.0, 1: 3.0, 2: 4.0}, average_edge_comm_s={(0, 1): 1.0, (1, 2): 1.0})

        self.assertEqual(ordered, (0, 1, 2))

    def test_core_weighted_costs_match_relay_example(self):
        runtime, ground, member, bs = make_runtime()
        costs = ERS(runtime).average_costs(DAGRequest(0, make_dag(), member, ground))

        self.assertEqual(set(costs.candidates), {ground, member, bs})
        self.assertAlmostEqual(costs.average_compute_s[0], (0.1 + 0.02 + 4 / 120) / 7)
        self.assertAlmostEqual(costs.average_edge_comm_s[(0, 1)], weighted_comm(runtime, ground, member, bs, 1e6))
        self.assertAlmostEqual(costs.path_seconds_per_bit[ground, bs],
                               hop_cost(runtime, ground, member) + hop_cost(runtime, member, bs))
        for node in costs.candidates:
            self.assertEqual(costs.path_seconds_per_bit[node, node], 0.)

    def test_asymmetric_routes_and_kbit_to_bits(self):
        runtime, ground, member, bs = make_runtime(asymmetric=True)
        costs = ERS(runtime).average_costs(DAGRequest(0, make_dag(edge_kbit=1.), member, ground))

        self.assertAlmostEqual(costs.path_seconds_per_bit[ground, bs],
                               hop_cost(runtime, ground, member) + hop_cost(runtime, member, bs))
        self.assertAlmostEqual(costs.path_seconds_per_bit[bs, ground],
                               hop_cost(runtime, bs, member) + hop_cost(runtime, member, ground))
        self.assertAlmostEqual(costs.average_edge_comm_s[(0, 1)], weighted_comm(runtime, ground, member, bs, 1000.))
        self.assertNotEqual(costs.path_seconds_per_bit[ground, bs], costs.path_seconds_per_bit[bs, ground])

    def test_input_size_does_not_change_ranks(self):
        runtime, ground, member, _ = make_runtime()
        ers = ERS(runtime)
        first = ers.plan([DAGRequest(0, make_dag(input_kbit=1.), member, ground)])
        large = ers.plan([DAGRequest(0, make_dag(input_kbit=1e6), member, ground)])
        self.assertEqual(first.ranks, large.ranks)
        self.assertEqual(first.order, large.order)
        self.assertFalse(runtime.tasks)
        self.assertFalse(runtime.channels)

    def test_multiple_dags_interleave_independently_of_request_order(self):
        runtime, ground, member, _ = make_runtime()
        a = DAGRequest(0, make_dag((4e8, 1e8)), member, ground)
        b = DAGRequest(1, make_dag((2e8, 2e8)), member, ground)
        plan = ERS(runtime).plan([a, b])
        expected = tuple(TaskKey(0, 0, dag, node) for dag, node in [(0, 0), (1, 0), (1, 1), (0, 1)])
        self.assertEqual(plan.order, expected)
        self.assertEqual(plan.member_orders[member], expected)
        self.assertEqual(ERS(runtime).plan([b, a]).order, expected)

    def test_candidates_exclude_other_users_and_other_regions(self):
        runtime, ground, member, bs = make_runtime()
        other_ground = EntityRef(EntityKind.GROUND_DEVICE, 1)
        remote = EntityRef(EntityKind.MEMBER_UAV, 1)
        positions = dict(runtime.entity_positions)
        positions.update({other_ground: np.array([500., 0., 0.]), remote: np.array([500., 0., 100.])})
        servers = {node: ServerSpec(s.core_frequencies, s.capacitance_factor) for node, s in runtime.servers.items()}
        servers.update({other_ground: ServerSpec((1e9,), 0.), remote: ServerSpec((1e9,) * 50, 1e-28)})
        extended = SchedulingRuntime(runtime.channel_model, positions, servers,
                                     {node: 1. for node in positions},
                                     member_regions={member: 1, remote: 2})
        costs = ERS(extended).average_costs(DAGRequest(0, make_dag(), member, ground))
        self.assertEqual(set(costs.candidates), {ground, member, bs})
        self.assertAlmostEqual(costs.average_edge_comm_s[(0, 1)], weighted_comm(extended, ground, member, bs, 1e6))
        requests = [DAGRequest(0, make_dag(), member, ground),
                    DAGRequest(0, make_dag(), remote, other_ground)]
        plan = ERS(extended).plan(requests)
        self.assertEqual(set(plan.costs[requests[1].key].candidates), {other_ground, remote, bs})
        for owner in (member, remote):
            self.assertEqual(plan.member_orders[owner],
                             tuple(key for key in plan.order if key.owner_member_id == owner.index))
        self.assertEqual(ERS(extended).plan(reversed(requests)).order, plan.order)

    def test_same_owner_users_keep_separate_routes_and_dag_identities(self):
        runtime, ground, member, bs = make_runtime()
        other_ground = EntityRef(EntityKind.GROUND_DEVICE, 1)
        positions = dict(runtime.entity_positions)
        positions[other_ground] = np.array([900., 0., 0.])
        servers = {node: ServerSpec(s.core_frequencies, s.capacitance_factor) for node, s in runtime.servers.items()}
        servers[other_ground] = ServerSpec((1e9,), 0.)
        extended = SchedulingRuntime(runtime.channel_model, positions, servers,
                                     {node: 1. for node in positions},
                                     member_regions={member: 1})
        requests = [DAGRequest(0, make_dag(), member, user) for user in (ground, other_ground)]
        plan = ERS(extended).plan(requests)
        self.assertEqual(len(plan.order), 4)
        self.assertEqual(plan.member_orders[member], plan.order)
        self.assertNotEqual(plan.ranks[requests[0].task_keys[0]], plan.ranks[requests[1].task_keys[0]])
        for request in requests:
            user = request.ground_device
            expected = (extended.transfer_duration_s(DirectedChannelKey(user, member), 1.)
                        + extended.transfer_duration_s(DirectedChannelKey(member, bs), 1.))
            self.assertAlmostEqual(plan.costs[request.key].path_seconds_per_bit[user, bs], expected)

    def test_zero_cost_ties_preserve_dependencies(self):
        dag = DAG(2, [(1, 0)], np.ones((2, 2)), {(1, 0): 0.})
        self.assertEqual(ERS.order(dag, {0: 0., 1: 0.}, {(1, 0): 0.}), (1, 0))

    def test_long_chain_does_not_need_python_recursion(self):
        dag = make_dag((1.,) * 1500)
        ranks = ERS.ranks(dag, {i: 1. for i in range(1500)}, {edge: 1. for edge in dag.edges})
        self.assertEqual(ranks[0], 2999.)
        self.assertEqual(ranks[1499], 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
