"""Independent frozen scheduler, ERS ordering, and paired episode evaluation.

Only environment modules are shared with the project. Lower actor tensors are
read from the pretrained checkpoint without importing any baseline policy.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

from env.env import MRDTSEnv
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.settings.uav_config import MemberPropulsionConfig
from env.runtime.dag_runtime import topological_order
from env.contracts import PlacementDecision
from env.types import EntityKind, EntityRef, TaskKey


def restore_environment(snapshot):
    settings = {}
    for key, factory in [('region', RegionConfig), ('user', UserConfig),
                         ('channel', ChannelConfig), ('dag', DAGConfig), ('env', EnvConfig)]:
        settings[key] = factory(**snapshot[key])
    aircraft = dict(snapshot['uav'])
    aircraft['member_propulsion'] = MemberPropulsionConfig(**aircraft['member_propulsion'])
    settings['uav'] = UAVConfig(**aircraft)
    environment = MRDTSEnv(settings)
    environment.generate_scene()
    return environment


def flight_actions(positions, targets, side, step):
    return np.clip((targets * side - positions) / step, -1, 1).astype(np.float32)


def task_orders(runtime, requests):
    """Core-weighted ERS ranks with per-owner stable topological tie breaking."""
    owners = {}
    contexts = {}
    hop_times = {}
    for request in sorted(requests, key=lambda r: r.key):
        if request.owner_member is None:
            continue
        context = request.owner_member, request.ground_device
        if context not in contexts:
            nodes = runtime.candidate_execution_nodes(*context)
            frequencies = {node: runtime.servers[node].core_frequencies for node in nodes}
            core_total = sum(len(values) for values in frequencies.values())
            compute_unit = math.fsum(1 / frequency for values in frequencies.values()
                                     for frequency in values) / core_total
            weighted_paths = []
            for start in nodes:
                for end in nodes:
                    route = runtime.route_planner.predecessor_route(start, end,
                                request.ground_device, request.owner_member)
                    for hop in route.hops:
                        if hop not in hop_times:
                            hop_times[hop] = runtime.transfer_duration_s(hop, 1.)
                    path_cost = math.fsum(hop_times[hop] for hop in route.hops)
                    weighted_paths.append(len(frequencies[start]) * len(frequencies[end]) * path_cost)
            contexts[context] = compute_unit, math.fsum(weighted_paths) / core_total ** 2
        compute_unit, communication_unit = contexts[context]
        dag = request.dag
        topology = topological_order(dag)
        successors = {node: [] for node in topology}
        for start, end in dag.edges:
            successors[start].append(end)
        rank = {}
        for node in reversed(topology):
            rank[node] = float(dag.node_features[node][1]) * compute_unit + max(
                (float(dag.edge_features[node, child]) * 1000 * communication_unit + rank[child]
                 for child in successors[node]), default=0.)
        tasks = owners.setdefault(request.owner_member.index, [])
        tasks.extend((-rank[node], request.ground_device.index, request.dag_id, index,
                      TaskKey(*request.key, node)) for index, node in enumerate(topology))
    return {owner: iter(item[-1] for item in sorted(tasks)) for owner, tasks in owners.items()}


class Simulator:
    def __init__(self, source):
        torch.set_num_threads(1)
        self.env = restore_environment(source['environment'])
        self.count = self.env.members.member_uav_count
        self.side = self.env.region.config.side_length
        self.step = self.env.members.config.max_horizontal_speed * self.env.env_config.flight_duration_s
        self.initial = self.env.members.positions[:self.count, :2].copy() / self.side
        self.observation = np.concatenate((self.initial.ravel(),
            self.env.users.positions[:, :2].ravel() / self.side)).astype(np.float32)
        weights = source['member_policies']
        self.actors = [tuple((weights[f'{owner}.actor.{layer}.weight'],
                             weights[f'{owner}.actor.{layer}.bias']) for layer in (0, 2, 4))
                       for owner in range(self.count)]
        self.member_nodes = tuple(EntityRef(EntityKind.MEMBER_UAV, i) for i in range(self.count))
        self.flight_seed = source['config']['flight_seed']
        self.resample = source['config']['resample_slots']

    def placement(self, key):
        observation = self.env.observe_member(key)
        start = 4 + 4 * self.count
        features = torch.from_numpy(np.concatenate((observation[:start], observation[start + self.count:]))).float()
        layers = self.actors[key.owner_member_id]
        for weight, bias in layers[:-1]:
            features = torch.tanh(F.linear(features, weight, bias))
        logits = F.linear(features, *layers[-1])
        regions = self.env.members.region_ids[:self.count]
        mask = torch.from_numpy(regions != regions[key.owner_member_id])
        logits[:self.count].masked_fill_(mask, -torch.inf)
        action = int(logits.argmax())
        if action < self.count:
            node = self.member_nodes[action]
        elif action == self.count:
            node = EntityRef(EntityKind.BS, 0)
        else:
            node = EntityRef(EntityKind.GROUND_DEVICE, key.user_id)
        return PlacementDecision(node)

    @torch.inference_mode()
    def evaluate(self, targets, epoch, slots, label='learned'):
        env = self.env
        env.reset(epoch_index=epoch)
        delay_sum, dags, failed, violations = 0., 0, 0, 0
        random = np.random.default_rng(np.random.SeedSequence([self.flight_seed, epoch]))
        actions = np.zeros((self.count, 2), dtype=np.float32)
        for slot in range(slots):
            positions = env.members.positions[:self.count, :2]
            if label == 'random':
                if slot % self.resample == 0:
                    actions = random.uniform(-1, 1, (self.count, 2)).astype(np.float32)
                proposed = positions + actions * self.step
                actions[(proposed < 0) | (proposed > self.side)] *= -1
            elif label != 'hover':
                actions = flight_actions(positions, targets, self.side, self.step)
            begin = env.begin_step(actions)
            violations += int(begin.flight_violations.sum())
            orders = task_orders(env._runtime, begin.requests)
            ready = begin.ready_member_ids
            while ready:
                keys = [next(orders[owner]) for owner in ready]
                ready = env.submit_schedule({key: self.placement(key) for key in keys})
                env.pop_member_rewards()
            result = env.end_step().slot_result
            for dag in result.dags:
                delay_sum += dag.completion_time - result.slot_start
                failed += dag.failed
            dags += len(result.dags)
        return dict(policy=label, epoch=epoch, slots_completed=slots, dag_count=dags,
                    mean_delay_s=delay_sum / dags, failed_dags=int(failed),
                    failure_rate=failed / dags, flight_violations=violations)


def initialize_worker(path):
    global SIMULATOR
    SIMULATOR = Simulator(torch.load(path, map_location='cpu', weights_only=True))


def evaluate_job(job):
    return SIMULATOR.evaluate(*job)
