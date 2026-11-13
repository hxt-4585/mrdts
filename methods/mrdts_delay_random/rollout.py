"""Roll out the regional flight policy against uniform random execution nodes."""
from collections import Counter
from dataclasses import asdict, replace
from itertools import zip_longest

import numpy as np
import torch

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.settings.uav_config import MemberPropulsionConfig
from env.types import EntityKind
from scripts.ers import ERS
from .high import FlightController


SETTINGS = dict(env=EnvConfig, channel=ChannelConfig, dag=DAGConfig,
                region=RegionConfig, user=UserConfig, uav=UAVConfig)


def environment_snapshot(train):
    settings = {name: cls.default() for name, cls in SETTINGS.items()}
    settings['env'] = replace(settings['env'], steps_per_epoch=train['slots'])
    if 'users' in train:
        settings['user'] = replace(settings['user'], total_users=train['users'])
    if 'dag_nodes' in train:
        settings['dag'] = replace(settings['dag'], n=train['dag_nodes'])
    return {name: asdict(value) for name, value in settings.items()}


def environment(snapshot):
    settings = dict(snapshot)
    settings['uav'] = dict(settings['uav'], member_propulsion=MemberPropulsionConfig(
        **settings['uav']['member_propulsion']))
    env = MRDTSEnv({name: cls(**settings[name]) for name, cls in SETTINGS.items()})
    env.generate_scene()
    return env


def random_schedule(runtime, requests, rng):
    plan = ERS(runtime).plan(requests)
    orders = [plan.member_orders[owner] for owner in sorted(plan.member_orders)]
    placements = {}
    for group in zip_longest(*orders):
        for key in group:
            if key is None:
                continue
            candidates = plan.costs[key.owner_member_id, key.user_id, key.dag_id].candidates
            node = candidates[rng.integers(len(candidates))]
            placements[key] = PlacementDecision(node)
    return placements


class Sampler:
    def __init__(self, snapshot, seed, index):
        self.env = environment(snapshot)
        self.seed = seed
        self.index = index

    def reset(self, epoch, delays):
        self.env.reset(epoch_index=epoch)
        self.flight = FlightController(self.env, delays, epoch)
        self.offloading = np.random.default_rng([self.seed, epoch, self.index, 200])

    def step(self, exploration, training):
        env = self.env
        scene = self.flight.observe(env)
        action = self.flight.act(scene, exploration)
        begin = env.begin_step(action)
        placements = random_schedule(env._runtime, begin.requests, self.offloading)
        env.submit_schedule(placements)
        result = env.end_step()
        slot = result.slot_result
        dags = slot.dags
        delays = np.array([dag.completion_time - slot.slot_start for dag in dags])
        failed = np.array([dag.failed for dag in dags], dtype=bool)
        counts = Counter(p.execution_node.kind for p in placements.values())
        row = dict(epoch=env.epoch_index, slot=env.step_index - 1, sampler=self.index,
                   dag_count=len(dags), delay_sum_s=float(delays.sum()), mean_delay_s=float(delays.mean()),
                   failed_dags=int(failed.sum()), failure_rate=float(failed.mean()),
                   successful_delay_sum_s=float(delays[~failed].sum()),
                   flight_violations=int(begin.flight_violations.sum()), member_decisions=len(placements),
                   ground_placements=counts[EntityKind.GROUND_DEVICE],
                   member_placements=counts[EntityKind.MEMBER_UAV], bs_placements=counts[EntityKind.BS],
                   global_delay_reward=result.master_reward)
        experience = None
        if training:
            regions = [env.users.region_ids[dag.key[1]] - 1 for dag in dags]
            labels = np.bincount(regions, weights=delays, minlength=len(scene.owned)) / len(dags)
            inputs = scene.encode(torch.from_numpy(scene.xy + action * self.flight.stride))
            experience = (inputs, labels)
        return row, experience
