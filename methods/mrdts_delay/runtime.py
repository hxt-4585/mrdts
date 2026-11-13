"""Environment construction, ERS ordering and whole-slot PPO rollouts."""
from dataclasses import replace
from itertools import zip_longest
import math

import numpy as np

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.runtime.dag_runtime import topological_order
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.types import EntityKind, EntityRef, TaskKey
from methods.mrdts_delay.high import regional_targets
from methods.mrdts_delay.low import advantages, member_input


def default_settings(train=None):
    constructors = dict(env=EnvConfig, region=RegionConfig, uav=UAVConfig,
                        user=UserConfig, channel=ChannelConfig, dag=DAGConfig)
    settings = {name: cls.default() for name, cls in constructors.items()}
    if train is not None:
        settings['env'] = replace(settings['env'], steps_per_epoch=train['slots'])
        if 'users' in train:
            settings['user'] = replace(settings['user'], total_users=train['users'])
        if 'dag_nodes' in train:
            settings['dag'] = replace(settings['dag'], n=train['dag_nodes'])
    return settings


def build_environment(settings):
    env = MRDTSEnv(settings)
    env.generate_scene()
    env.reset(epoch_index=0)
    return env


def execution_order(runtime, requests):
    """Core-weighted upward ranks; topological positions resolve equal-rank ties."""
    estimates, hop_times, ranking = {}, {}, []
    for request in sorted(requests, key=lambda item: item.key):
        owner = request.owner_member
        if owner is None:
            continue
        context = (owner, request.ground_device)
        if context not in estimates:
            nodes = runtime.candidate_execution_nodes(*context)
            cores = [runtime.servers[node].core_frequencies for node in nodes]
            count = sum(len(group) for group in cores)
            compute = math.fsum(1 / frequency for group in cores for frequency in group) / count
            communication = 0.
            for i, source in enumerate(nodes):
                for j, destination in enumerate(nodes):
                    route = runtime.route_planner.predecessor_route(source, destination,
                                                                    request.ground_device, owner)
                    duration = 0.
                    for hop in route.hops:
                        if hop not in hop_times:
                            hop_times[hop] = runtime.transfer_duration_s(hop, 1.)
                        duration += hop_times[hop]
                    communication += duration * len(cores[i]) * len(cores[j])
            estimates[context] = (compute, communication / count ** 2)
        compute, communication = estimates[context]
        dag = request.dag
        topology = topological_order(dag)
        outgoing = [[] for _ in range(dag.node_num)]
        for parent, child in dag.edges:
            outgoing[parent].append(child)
        ranks = np.empty(dag.node_num)
        for node in reversed(topology):
            ranks[node] = dag.node_features[node, 1] * compute + max(
                (ranks[child] + dag.edge_features[node, child] * 1000 * communication
                 for child in outgoing[node]), default=0.)
        for position, node in enumerate(topology):
            ranking.append((-ranks[node], *request.key, position, node))
    orders = {}
    for _, owner, user, dag, _, node in sorted(ranking):
        orders.setdefault(owner, []).append(TaskKey(owner, user, dag, node))
    # Interleave local ERS sequences in Member ID order for shared-resource ties.
    rounds = zip_longest(*(orders[owner] for owner in sorted(orders)))
    return (key for group in rounds for key in group if key is not None)


def run_slot(env, low, actions, config, deterministic=False):
    count = env.members.member_uav_count
    trajectories = [[] for _ in range(count)]
    begin = env.begin_step(actions)
    regions = env.members.region_ids[:count]
    masks = np.ones((count, count + 2), dtype=bool)
    masks[:, :count] = regions[:, None] == regions[None, :]
    placements = {}
    samples = {}
    for key in execution_order(env._runtime, begin.requests):
        member = key.owner_member_id
        observation = member_input(env.observe_member(key, placements), count)
        sample = low.act(member, observation, masks[member], deterministic)
        trajectories[member].append(sample)
        samples[key] = sample
        if sample.action < count:
            node = EntityRef(EntityKind.MEMBER_UAV, sample.action)
        elif sample.action == count:
            node = EntityRef(EntityKind.BS, 0)
        else:
            node = EntityRef(EntityKind.GROUND_DEVICE, key.user_id)
        placements[key] = PlacementDecision(node)

    env.submit_schedule(placements)
    result = env.end_step()
    for key, sample in samples.items():
        sample.reward = result.member_rewards[key].compute_delay()
    for trajectory in trajectories:
        advantages(trajectory, config['member_gamma'], config['gae_lambda'])
    slot = result.slot_result
    targets = regional_targets(slot, env.users.region_ids, env.region.config.region_count)
    delay = float(sum(dag.completion_time - slot.slot_start for dag in slot.dags))
    failed = int(sum(dag.failed for dag in slot.dags))
    successful = float(sum(dag.completion_time - slot.slot_start for dag in slot.dags if not dag.failed))
    n = len(slot.dags)
    row = dict(epoch=env.epoch_index, slot=env.step_index - 1, dag_count=n,
               delay_sum_s=delay, mean_delay_s=delay / n, failed_dags=failed, failure_rate=failed / n,
               successful_delay_sum_s=successful,
               successful_mean_delay_s=successful / (n - failed) if n > failed else None,
               flight_violations=int(begin.flight_violations.sum()),
               member_decisions=sum(len(seq) for seq in trajectories),
               member_reward_sum=sum(s.reward for seq in trajectories for s in seq),
               global_delay_reward=result.master_reward, total_energy_j=float(result.total_energy_j.sum()),
               mean_energy_queue_j=float(result.energy_virtual_queue_j.mean()))
    return trajectories, row, targets
