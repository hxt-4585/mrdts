"""Whole-window, region-constrained HEFT using exclusive-link cost estimates.

Ranks average over eligible CPU cores. EFT includes every task's original
input, predecessor routes and shared per-core insertion calendars. Calendars
are predictions only: env chooses actual cores and ready-time queue order,
and dynamically shares bandwidth/power. No reservation is passed to env.

Reference: Topcuoglu et al. (2002), doi:10.1109/71.993206.
"""
from dataclasses import dataclass
import heapq
import math

from env.contracts import PlacementDecision
from env.runtime.dag_runtime import topological_order
from env.types import TaskKey


@dataclass(frozen=True)
class Estimate:
    core: int
    start: float
    finish: float


@dataclass
class Schedule:
    placements: dict
    tasks: dict[TaskKey, Estimate]
    ranks: dict[TaskKey, float]


def earliest_gap(calendar, release, duration):
    """Return earliest feasible start and insertion index in a sorted calendar."""
    start = release
    for index, (occupied_start, occupied_finish) in enumerate(calendar):
        if start + duration <= occupied_start:
            return start, index
        start = max(start, occupied_finish)
    return start, len(calendar)


def plan_schedule(runtime, requests):
    """Plan all associated DAGs jointly without submitting or advancing events."""
    ranks, contexts, parents, children, pending = {}, {}, {}, {}, {}
    unit_hops = {}
    for request in sorted(requests, key=lambda item: item.key):
        if request.owner_member is None:
            continue
        nodes = runtime.candidate_execution_nodes(request.owner_member, request.ground_device)
        cores = [(node, core, frequency) for node in nodes
                 for core, frequency in enumerate(runtime.servers[node].core_frequencies)]

        def route_cost(route):
            for hop in route.hops:
                if hop not in unit_hops:
                    unit_hops[hop] = runtime.transfer_duration_s(hop, 1.)
            return math.fsum(unit_hops[hop] for hop in route.hops)

        inputs = {node: route_cost(runtime.route_planner.input_route(
            request.ground_device, request.owner_member, node)) for node in nodes}
        edges = {(source, target): route_cost(runtime.route_planner.predecessor_route(
            source, target, request.ground_device, request.owner_member))
            for source in nodes for target in nodes}
        weights = {node: len(runtime.servers[node].core_frequencies) for node in nodes}
        average_compute = math.fsum(1 / frequency for _, _, frequency in cores) / len(cores)
        average_transfer = math.fsum(cost * weights[source] * weights[target]
                                     for (source, target), cost in edges.items()) / len(cores) ** 2
        keys = request.task_keys
        for key in keys:
            parents[key], children[key] = [], []
            contexts[key] = (cores, inputs, edges)
        for parent, child in request.dag.edges:
            bits = float(request.dag.edge_features[parent, child]) * 1000.
            parents[keys[child]].append((keys[parent], bits))
            children[keys[parent]].append(keys[child])
        for node in reversed(topological_order(request.dag)):
            key = keys[node]
            ranks[key] = runtime.tasks[key].cpu_cycles * average_compute + max(
                (ranks[child] + bits * average_transfer
                 for child, bits in ((keys[c], request.dag.edge_features[node, c] * 1000.)
                                     for c in runtime.dag_runtimes[request.key].successors[node])),
                default=0.)
        for key in keys:
            pending[key] = len(parents[key])

    ready = [(-ranks[key], key) for key, degree in pending.items() if degree == 0]
    heapq.heapify(ready)
    calendars = {(node, core): [] for node, server in runtime.servers.items()
                 for core in range(len(server.core_frequencies))}
    placements, estimates = {}, {}
    while ready:
        _, key = heapq.heappop(ready)
        task = runtime.tasks[key]
        cores, input_costs, edge_costs = contexts[key]
        releases = {}
        for node in input_costs:
            releases[node] = max(runtime.slot_start + task.input_bits * input_costs[node],
                                 max((estimates[parent].finish + bits * edge_costs[
                                     placements[parent].execution_node, node]
                                      for parent, bits in parents[key]), default=runtime.slot_start))
        best = None
        for ordinal, (node, core, frequency) in enumerate(cores):
            duration = task.cpu_cycles / frequency
            start, index = earliest_gap(calendars[node, core], releases[node], duration)
            option = (start + duration, start, ordinal, index)
            if best is None or option < best:
                best = option
        finish, start, ordinal, index = best
        node, core, _ = cores[ordinal]
        if finish > start:
            calendars[node, core].insert(index, (start, finish))
        placements[key] = PlacementDecision(node)
        estimates[key] = Estimate(core, start, finish)
        for child in children[key]:
            pending[child] -= 1
            if pending[child] == 0:
                heapq.heappush(ready, (-ranks[child], child))
    return Schedule(placements, estimates, ranks)
