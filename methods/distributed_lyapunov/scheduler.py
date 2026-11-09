"""Lyapunov ERS, with core-weighted compute and routed UAV transmit costs.

Q is frozen at the beginning of a slot. Ground devices and BS have no Q.
Reference: JTPDTS.tex, Lyapunov-Guided Dependent Task Scheduling.
"""
import math

from env.runtime.dag_runtime import topological_order
from env.types import EntityKind, TaskKey


def route_cost(runtime, route, queues, v):
    return math.fsum(runtime.transfer_duration_s(hop, 1.) *
                     (v + (queues[hop.source.index] * runtime.transmit_powers[hop.source]
                           if hop.source.kind is EntityKind.MEMBER_UAV else 0.))
                     for hop in route.hops)


def rank_orders(env, requests, queues, v):
    runtime = env._runtime
    coefficients, orders = {}, {}
    for request in requests:
        if request.owner_member is None:
            continue
        context = request.owner_member, request.ground_device
        if context not in coefficients:
            nodes = runtime.candidate_execution_nodes(*context)
            cores = {node: runtime.servers[node].core_frequencies for node in nodes}
            total = sum(map(len, cores.values()))
            comp = math.fsum(v / f + (queues[node.index] * runtime.servers[node].capacitance_factor * f**2
                                     if node.kind is EntityKind.MEMBER_UAV else 0.)
                             for node in nodes for f in cores[node]) / total
            comm = math.fsum(len(cores[a]) * len(cores[b]) * route_cost(runtime,
                            runtime.route_planner.predecessor_route(a, b, request.ground_device,
                                                                   request.owner_member), queues, v)
                            for a in nodes for b in nodes) / total**2
            coefficients[context] = comp, comm
        comp, comm = coefficients[context]
        dag = request.dag
        topology = topological_order(dag)
        children = {node: [] for node in topology}
        for a, b in dag.edges:
            children[a].append(b)
        ranks = {}
        for node in reversed(topology):
            ranks[node] = float(dag.node_features[node][1]) * comp + max(
                (1000. * dag.edge_features[node, child] * comm + ranks[child]
                 for child in children[node]), default=0.)
        order = orders.setdefault(request.owner_member.index, [])
        order.extend((-ranks[node], request.ground_device.index, request.dag_id, position,
                      TaskKey(*request.key, node)) for position, node in enumerate(topology))
    return {owner: [entry[-1] for entry in sorted(order)] for owner, order in orders.items()}
