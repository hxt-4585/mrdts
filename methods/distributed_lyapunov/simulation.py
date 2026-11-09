"""Unmodified environment, exact slot energy accounting and continuous queue QA."""
from dataclasses import replace
import numpy as np
from env.env import MRDTSEnv
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.contracts import PlacementDecision
from env.types import EntityKind, EntityRef
from .learning import MemberPolicy
from .scheduler import rank_orders


def build_env(budget):
    settings = dict(env=EnvConfig.default(), region=RegionConfig.default(), user=UserConfig.default(),
                    uav=replace(UAVConfig.default(), member_slot_energy=budget),
                    channel=ChannelConfig.default(), dag=DAGConfig.default())
    env = MRDTSEnv(settings)
    env.generate_scene()
    return env


def objective(delay, queues, energy, v):
    return float(v * delay + np.dot(queues, energy))


def queue_statistics(queues, energies, budget):
    q, e = np.asarray(queues), np.asarray(energies)
    tail = q[len(q)//2:]
    t = np.arange(len(tail), dtype=float)
    centered = t - t.mean()
    slope = centered @ tail / max(float(centered @ centered), 1.)
    excess = e.mean(0) - budget
    # Finite-horizon empirical criterion; never a proof of strong stability.
    stable = bool(np.all(slope <= .001 * budget) and np.all(excess <= .005 * budget)
                  and np.all(q[-1] / len(q) <= .01 * budget))
    return dict(empirically_stable=stable, mean_queue_j=q.mean(0).tolist(),
                max_queue_j=q.max(0).tolist(), final_queue_j=q[-1].tolist(),
                tail_slope_j_per_slot=slope.tolist(), mean_energy_j=e.mean(0).tolist(),
                energy_excess_j=excess.tolist(), final_queue_over_horizon=(q[-1]/len(q)).tolist())


def execute(env, policies, actions, v, deterministic=False, heuristic=False):
    count = env.members.member_uav_count
    q0 = env.members.energy_virtual_queue_j.copy()
    begin = env.begin_step(actions)
    orders = {owner: iter(order) for owner, order in rank_orders(env, begin.requests, q0, v).items()}
    records = [[] for _ in range(count)]
    active = begin.ready_member_ids
    while active:
        decisions = {}
        for owner in active:
            key = next(orders[owner])
            obs, mask = MemberPolicy.inputs(env, key)
            if heuristic:
                action = heuristic_action(env, key, q0, v, mask)
                logp = value = 0.
            else:
                action, logp, value = policies[owner].choose(obs, mask, deterministic)
            records[owner].append([obs, mask, action, logp, value])
            target = (EntityRef(EntityKind.MEMBER_UAV, action) if action < count else
                      EntityRef(EntityKind.BS, 0) if action == count else
                      EntityRef(EntityKind.GROUND_DEVICE, key.user_id))
            decisions[key] = PlacementDecision(target)
        active = env.submit_schedule(decisions)
        env.pop_member_rewards()
    end = env.end_step()
    result = end.slot_result
    regional = np.zeros(env.region.config.region_count)
    for dag in result.dags:
        regional[env.users.region_ids[dag.key[1]] - 1] += dag.completion_time - result.slot_start
    costs = v * regional
    for owner, region in enumerate(env.members.region_ids[:count]):
        costs[region - 1] += q0[owner] * end.total_energy_j[owner]
    cost = objective(float(regional.sum()), q0, end.total_energy_j, v)
    # Shared one-slot team objective gives every sampled placement its joint
    # policy-gradient cost, including hover and UAV relay externalities.
    reward = -cost / (v * env.users.num_users)
    for trajectory in records:
        for record in trajectory:
            record.append(reward)
    row = dict(delay_sum_s=float(regional.sum()), mean_delay_s=float(regional.sum()/len(result.dags)),
               dag_count=len(result.dags), failure_rate=sum(d.failed for d in result.dags)/len(result.dags),
               objective=cost, violations=int(begin.flight_violations.sum()),
               energy_j=end.total_energy_j.tolist(), queue_j=end.energy_virtual_queue_j.tolist(),
               hover_energy_j=end.hover_energy_j.tolist(), flight_energy_j=end.flight_energy_j.tolist(),
               compute_energy_j=end.compute_energy_j.tolist(), tx_energy_j=end.tx_energy_j.tolist())
    return records, row, costs


def heuristic_action(env, key, q, v, mask):
    """Pilot-only WFT approximation, not used to label PPO training data."""
    from .scheduler import route_cost
    runtime = env._runtime
    task = runtime.tasks[key]
    ground = EntityRef(EntityKind.GROUND_DEVICE, key.user_id)
    owner = EntityRef(EntityKind.MEMBER_UAV, key.owner_member_id)
    count = env.members.member_uav_count
    scores = []
    for action in np.flatnonzero(mask):
        node = (EntityRef(EntityKind.MEMBER_UAV, int(action)) if action < count else
                EntityRef(EntityKind.BS, 0) if action == count else ground)
        server = runtime.servers[node]
        f = server.core_frequencies[0]
        value = v * (task.cpu_cycles/f + server.compute_wait(runtime.now))
        if node.kind is EntityKind.MEMBER_UAV:
            value += q[node.index] * server.capacitance_factor * task.cpu_cycles * f**2
        value += task.input_bits * route_cost(runtime, runtime.route_planner.input_route(ground, owner, node), q, v)
        request = runtime._requests[key.owner_member_id, key.user_id, key.dag_id]
        for parent in task.predecessors:
            bits = request.dag.edge_features[parent.node_id, key.node_id] * 1000.
            value += bits * route_cost(runtime, runtime.route_planner.predecessor_route(
                runtime.tasks[parent].execution_node, node, ground, owner), q, v)
        scores.append((value, int(action)))
    return min(scores)[1]
