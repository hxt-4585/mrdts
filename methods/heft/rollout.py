"""Environment setup and HEFT rollouts without dependencies on other baselines."""
from collections import Counter
from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
import torch

from env.env import MRDTSEnv
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.settings.uav_config import MemberPropulsionConfig
from env.types import EntityKind
from .scheduler import plan_schedule


CONFIG_TYPES = dict(env=EnvConfig, region=RegionConfig, uav=UAVConfig,
                    user=UserConfig, channel=ChannelConfig, dag=DAGConfig)


def make_settings(config):
    settings = {name: cls.default() for name, cls in CONFIG_TYPES.items()}
    settings['env'] = replace(settings['env'], steps_per_epoch=config['slots'])
    if 'users' in config:
        settings['user'] = replace(settings['user'], total_users=config['users'])
    if 'dag_nodes' in config:
        settings['dag'] = replace(settings['dag'], n=config['dag_nodes'])
    return {name: asdict(value) for name, value in settings.items()}


def make_environment(snapshot):
    values = dict(snapshot)
    values['uav'] = dict(values['uav'])
    values['uav']['member_propulsion'] = MemberPropulsionConfig(**values['uav']['member_propulsion'])
    env = MRDTSEnv({name: cls(**values[name]) for name, cls in CONFIG_TYPES.items()})
    env.generate_scene()
    return env


def run_slot(env, policy, exploration=0.):
    scene = policy.observe(env)
    actions = policy.act(scene, exploration)
    begin = env.begin_step(actions)
    clock = perf_counter()
    plan = plan_schedule(env._runtime, begin.requests)
    planning_seconds = perf_counter() - clock
    env.submit_schedule(plan.placements)
    outcome = env.end_step()
    slot = outcome.slot_result
    delays = np.array([dag.completion_time - slot.slot_start for dag in slot.dags])
    labels = np.zeros(len(policy.region_ids), dtype=np.float32)
    for dag, delay in zip(slot.dags, delays):
        labels[env.users.region_ids[dag.key[1]] - 1] += delay / len(delays)
    # Encode actual post-flight positions with pre-flight visibility/ownership.
    positions = torch.tensor((env.members.positions[:policy.member_count, :2] / policy.side)
                             .astype(np.float32))
    experience = (scene.features(positions), labels)
    counts = Counter(value.execution_node.kind for value in plan.placements.values())
    failed = sum(dag.failed for dag in slot.dags)
    predicted = max((task.finish - slot.slot_start for task in plan.tasks.values()), default=0.)
    row = dict(epoch=env.epoch_index, slot=env.step_index - 1, dag_count=len(delays),
               delay_sum_s=float(delays.sum()), mean_delay_s=float(delays.mean()),
               failed_dags=failed, failure_rate=failed / len(delays),
               successful_delay_sum_s=float(sum(delay for dag, delay in zip(slot.dags, delays) if not dag.failed)),
               member_decisions=len(plan.placements), ground_placements=counts[EntityKind.GROUND_DEVICE],
               member_placements=counts[EntityKind.MEMBER_UAV], bs_placements=counts[EntityKind.BS],
               flight_violations=int(begin.flight_violations.sum()),
               global_delay_reward=outcome.master_reward, total_energy_j=float(outcome.total_energy_j.sum()),
               mean_energy_queue_j=float(outcome.energy_virtual_queue_j.mean()),
               predicted_associated_makespan_s=predicted,
               actual_makespan_s=float(delays.max()), planning_seconds=planning_seconds)
    return row, experience
