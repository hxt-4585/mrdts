"""Standalone ERS implementation: core-weighted costs and stable upward ranks."""

import math

from env.runtime.dag_runtime import topological_order
from env.types import TaskKey


def order_requests(runtime, requests):
    ranks, positions, owners = {}, {}, set()
    contexts, hops = {}, {}
    for request in sorted(requests, key=lambda r: r.key):
        if request.owner_member is None:
            continue
        owner, ground = request.owner_member, request.ground_device
        owners.add(owner.index)
        context = owner, ground
        if context not in contexts:
            nodes = runtime.candidate_execution_nodes(owner, ground)
            cores = {node: runtime.servers[node].core_frequencies for node in nodes}
            count = sum(map(len, cores.values()))
            compute_cost = math.fsum(1 / f for fs in cores.values() for f in fs) / count
            costs = []
            for source in nodes:
                for target in nodes:
                    route = runtime.route_planner.predecessor_route(source, target, ground, owner)
                    for hop in route.hops:
                        if hop not in hops:
                            hops[hop] = runtime.transfer_duration_s(hop, 1.)
                    costs.append(len(cores[source]) * len(cores[target]) *
                                 math.fsum(hops[hop] for hop in route.hops))
            contexts[context] = compute_cost, math.fsum(costs) / count**2
        compute_cost, communication_cost = contexts[context]
        dag = request.dag
        topology = topological_order(dag)
        successors = {node: [] for node in topology}
        for parent, child in dag.edges:
            successors[parent].append(child)
        local = {}
        for node in reversed(topology):
            local[node] = float(dag.node_features[node, 1]) * compute_cost + max(
                (float(dag.edge_features[node, child]) * 1000 * communication_cost + local[child]
                 for child in successors[node]), default=0.)
        for position, node in enumerate(topology):
            key = TaskKey(*request.key, node)
            ranks[key], positions[key] = local[node], position
    ordered = sorted(ranks, key=lambda k: (-ranks[k], k.owner_member_id, k.user_id, k.dag_id, positions[k]))
    return {owner: tuple(key for key in ordered if key.owner_member_id == owner) for owner in sorted(owners)}
