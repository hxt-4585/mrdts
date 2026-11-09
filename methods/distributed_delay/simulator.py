"""Environment adapter and independently implemented frozen ERS executor.

No policy, scheduler or evaluator is imported from another baseline.
"""

from dataclasses import fields
import math
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from env.env import MRDTSEnv
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.settings.uav_config import MemberPropulsionConfig
from env.runtime.dag_runtime import topological_order
from env.contracts import PlacementDecision
from env.types import EntityKind, EntityRef, TaskKey


def environment_from_snapshot(snapshot):
    constructors = dict(env=EnvConfig, region=RegionConfig, user=UserConfig,
                        channel=ChannelConfig, dag=DAGConfig, uav=UAVConfig)
    settings = {}
    for name, constructor in constructors.items():
        saved = snapshot[name]
        values = {field.name: saved[field.name] for field in fields(constructor) if field.name in saved}
        if name == 'uav':
            values['member_propulsion'] = MemberPropulsionConfig(**values['member_propulsion'])
        settings[name] = constructor(**values)
    env = MRDTSEnv(settings)
    env.generate_scene()
    return env


def ers_queues(runtime, requests):
    """Upward ranks using core-weighted compute and routed communication costs."""
    queues, cache, hops = {}, {}, {}
    for request in requests:
        if request.owner_member is None:
            continue
        context = request.owner_member, request.ground_device
        if context not in cache:
            nodes = runtime.candidate_execution_nodes(*context)
            cores = [runtime.servers[node].core_frequencies for node in nodes]
            total = sum(map(len, cores))
            compute = math.fsum(1. / frequency for frequencies in cores for frequency in frequencies) / total
            communications = []
            for p, source in enumerate(nodes):
                for q, destination in enumerate(nodes):
                    path = runtime.route_planner.predecessor_route(
                        source, destination, request.ground_device, request.owner_member)
                    for hop in path.hops:
                        if hop not in hops:
                            hops[hop] = runtime.transfer_duration_s(hop, 1.)
                    communications.append(len(cores[p]) * len(cores[q]) *
                                          math.fsum(hops[hop] for hop in path.hops))
            cache[context] = compute, math.fsum(communications) / (total * total)
        compute, communication = cache[context]
        dag = request.dag
        topology = topological_order(dag)
        children = {node: [] for node in topology}
        for parent, child in dag.edges:
            children[parent].append(child)
        ranks = {}
        for node in reversed(topology):
            following = [dag.edge_features[node, child] * 1000. * communication + ranks[child]
                         for child in children[node]]
            ranks[node] = float(dag.node_features[node][1]) * compute + max(following, default=0.)
        owner = request.owner_member.index
        queue = queues.setdefault(owner, [])
        for position, node in enumerate(topology):
            key = TaskKey(*request.key, node)
            queue.append((-ranks[node], key.user_id, key.dag_id, position, key))
    return {owner: iter(entry[-1] for entry in sorted(queue)) for owner, queue in queues.items()}


class FrozenSimulation:
    def __init__(self, source):
        self.env = environment_from_snapshot(source['environment'])
        self.count = self.env.members.member_uav_count
        self.side = self.env.region.config.side_length
        self.distance = self.env.env_config.flight_duration_s * self.env.members.config.max_horizontal_speed
        tensors = source['member_policies']
        self.layers = []
        for member in range(self.count):
            self.layers.append([(tensors[f'{member}.actor.{index}.weight'].detach(),
                                 tensors[f'{member}.actor.{index}.bias'].detach()) for index in (0, 2, 4)])
        self.nodes = [EntityRef(EntityKind.MEMBER_UAV, member) for member in range(self.count)]
        self.user_regions = self.env.users.region_ids.copy()
        self.region_count = self.env.region.config.region_count

    def decision(self, key):
        raw = self.env.observe_member(key)
        split = 4 + 4 * self.count
        observation = torch.from_numpy(np.delete(raw, np.s_[split:split + self.count])).float()
        first, second, last = self.layers[key.owner_member_id]
        hidden = torch.tanh(F.linear(observation, *first))
        hidden = torch.tanh(F.linear(hidden, *second))
        scores = F.linear(hidden, *last)
        membership = self.env.members.region_ids[:self.count]
        excluded = torch.from_numpy(membership != membership[key.owner_member_id])
        scores[:self.count].masked_fill_(excluded, -torch.inf)
        chosen = int(scores.argmax())
        endpoints = self.nodes + [EntityRef(EntityKind.BS, 0), EntityRef(EntityKind.GROUND_DEVICE, key.user_id)]
        return PlacementDecision(endpoints[chosen])

    @torch.no_grad()
    def step(self, actions):
        begin = self.env.begin_step(actions)
        queues = ers_queues(self.env._runtime, begin.requests)
        active = begin.ready_member_ids
        while active:
            batch = [next(queues[member]) for member in active]
            active = self.env.submit_schedule({key: self.decision(key) for key in batch})
            self.env.pop_member_rewards()
        slot = self.env.end_step().slot_result
        regional = np.zeros(self.region_count, dtype=np.float64)
        for dag in slot.dags:
            regional[self.user_regions[dag.key[1]] - 1] += dag.completion_time - slot.slot_start
        return dict(delay_sum_s=float(regional.sum()), dag_count=len(slot.dags),
                    failed_dags=sum(dag.failed for dag in slot.dags),
                    flight_violations=int(begin.flight_violations.sum())), regional

    def episode(self, epoch, slots, *, targets=None, controller=None, learn=False, exploration=0.):
        from methods.distributed_delay.planner import frames_from_env
        self.env.reset(epoch_index=epoch)
        totals = dict(delay_sum_s=0., dag_count=0, failed_dags=0, flight_violations=0)
        planning, timings, communication, evaluations, losses = 0., [], 0, 0, []
        action_messages = gradient_messages = 0
        if controller is not None:
            controller.reset(epoch)
        for slot in range(slots):
            started = perf_counter()
            if targets is not None:
                actions = np.clip((targets * self.side - self.env.members.positions[:self.count, :2]) /
                                  self.distance, -1., 1.).astype(np.float32)
                stats = dict(payload_scalars=0, proxy_evaluations=0, action_messages=0, gradient_messages=0)
            else:
                frames = frames_from_env(self.env)
                actions, stats = controller.actions(frames, exploration=exploration)
            elapsed = perf_counter() - started
            planning += elapsed
            timings.append(elapsed)
            communication += stats['payload_scalars']
            evaluations += stats['proxy_evaluations']
            action_messages += stats['action_messages']
            gradient_messages += stats['gradient_messages']
            row, regional = self.step(actions)
            for key in totals:
                totals[key] += row[key]
            if learn:
                loss = controller.feedback(frames, actions, regional)
                losses.append(loss)
        return dict(epoch=epoch, slots_completed=slots, **totals,
                    mean_delay_s=totals['delay_sum_s'] / totals['dag_count'],
                    failure_rate=totals['failed_dags'] / totals['dag_count'],
                    planning_mean_s=planning / slots, planning_p95_s=float(np.quantile(timings, .95)),
                    neighbor_payload_scalars=communication, proxy_evaluations=evaluations,
                    action_messages=action_messages, gradient_messages=gradient_messages,
                    proxy_loss=float(np.mean(losses)) if losses else None)
