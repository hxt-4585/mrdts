"""Scene snapshots and metrics; dependencies are limited to the shared environment."""

from dataclasses import replace
import json
from pathlib import Path

from env.env import MRDTSEnv
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.settings.uav_config import MemberPropulsionConfig

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = dict(region=RegionConfig, user=UserConfig, uav=UAVConfig,
                channel=ChannelConfig, dag=DAGConfig, env=EnvConfig)


def resolved_settings(users=None, dag_nodes=None):
    result = {name: cls.default() for name, cls in SETTINGS.items()}
    if users is not None:
        result['user'] = replace(result['user'], total_users=users)
    if dag_nodes is not None:
        result['dag'] = replace(result['dag'], n=dag_nodes)
    return result


def settings_from_snapshot(snapshot):
    result = {}
    for name, cls in SETTINGS.items():
        fields = dict(snapshot[name])
        if name == 'uav':
            fields['member_propulsion'] = MemberPropulsionConfig(**fields['member_propulsion'])
        result[name] = cls(**fields)
    return result


def build_scene(settings, *, episode):
    env = MRDTSEnv(settings)
    env.generate_scene()
    env.reset(epoch_index=episode)
    return env


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


class Summary:
    def __init__(self):
        self.slots = self.dags = self.failed = self.decisions = self.violations = 0
        self.delay = self.reward = self.global_reward = self.energy = 0.

    def add(self, row):
        self.slots += 1
        self.dags += row['dag_count']
        self.failed += row['failed_dags']
        self.decisions += row['member_decisions']
        self.delay += row['delay_sum_s']
        self.reward += row['member_reward_sum']
        self.global_reward += row['global_delay_reward']
        self.energy += row['total_energy_j']
        self.violations += row['flight_violations']

    def result(self):
        return dict(slots_completed=self.slots, dag_count=self.dags, failed_dags=self.failed,
                    mean_delay_s=self.delay / self.dags if self.dags else None,
                    failure_rate=self.failed / self.dags if self.dags else None,
                    member_decisions=self.decisions, member_reward_sum=self.reward,
                    member_reward_per_decision=self.reward / self.decisions if self.decisions else None,
                    global_delay_reward_mean=self.global_reward / self.slots if self.slots else None,
                    total_energy_j=self.energy, flight_violations=self.violations)
