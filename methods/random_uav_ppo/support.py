"""随机飞行基线的场景初始化、指标统计和结果文件辅助函数。"""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess

from env.env import MRDTSEnv
from env.settings.uav_config import MemberPropulsionConfig
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolved_settings(users=None, dag_nodes=None):
    settings = dict(region=RegionConfig.default(), user=UserConfig.default(),
                    uav=UAVConfig.default(), channel=ChannelConfig.default(),
                    dag=DAGConfig.default(), env=EnvConfig.default())
    if users is not None:
        settings['user'] = replace(settings['user'], total_users=users)
    if dag_nodes is not None:
        settings['dag'] = replace(settings['dag'], n=dag_nodes)
    return settings


def build_scene(settings, *, episode):
    """场景和本 episode 的任务均由环境配置中的 env_seed 控制。"""
    environment = MRDTSEnv(settings)
    environment.generate_scene()
    environment.reset(epoch_index=episode)
    return environment


def slot_metrics(result, episode, slot, flight_violations):
    count = len(result.dags)
    failed = 0
    delay_sum = successful_delay_sum = 0.
    for dag in result.dags:
        delay = dag.completion_time - result.slot_start
        delay_sum += delay
        failed += dag.failed
        if not dag.failed:
            successful_delay_sum += delay
    finished = count - failed
    return dict(episode=episode, slot=slot, dag_count=count, failed_dags=failed,
                failure_rate=failed / count if count else 0.,
                successful_delay_sum_s=successful_delay_sum,
                successful_mean_delay_s=successful_delay_sum / finished if finished else None,
                delay_sum_s=delay_sum,
                mean_delay_s=delay_sum / count if count else None,
                flight_violations=flight_violations)


def summarize(rows):
    summary = MetricSummary()
    for row in rows:
        summary.add(row)
    return summary.result()


@dataclass
class BaseSummary:
    """常量内存汇总；时延按 DAG 加权，奖励按 slot 平均。"""

    slots: int = 0
    dags: int = 0
    failed: int = 0
    delay: float = 0.
    successful_delay: float = 0.
    violations: int = 0
    global_delay_reward: float = 0.
    member_reward: float = 0.
    member_decisions: int = 0

    def add(self, row):
        self.slots += 1
        self.dags += row['dag_count']
        self.failed += row['failed_dags']
        self.delay += row['delay_sum_s']
        self.successful_delay += row['successful_delay_sum_s']
        self.violations += row['flight_violations']
        self.global_delay_reward += row['global_delay_reward']
        self.member_reward += row['member_reward_sum']
        self.member_decisions += row['member_decisions']

    def result(self):
        finished = self.dags - self.failed
        return dict(slots_completed=self.slots, dag_count=self.dags, failed_dags=self.failed,
                    failure_rate=self.failed / self.dags if self.dags else 0.,
                    successful_mean_delay_s=self.successful_delay / finished if finished else None,
                    mean_delay_s=self.delay / self.dags if self.dags else None,
                    flight_violations=self.violations,
                    global_delay_reward_mean=self.global_delay_reward / self.slots if self.slots else None,
                    member_reward_sum=self.member_reward, member_decisions=self.member_decisions,
                    member_reward_per_decision=(self.member_reward / self.member_decisions
                                                if self.member_decisions else None))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def git_output(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class MetricSummary(BaseSummary):
    def __init__(self):
        super().__init__()
        self.crossings = self.away_slots = self.away_decisions = self.empty_slots = 0
        self.away_reward = 0.
        self.away_system = BaseSummary()

    def add(self, row):
        super().add(row)
        self.crossings += row['flight_region_crossings']
        self.away_slots += row['flight_away_members'] > 0
        self.empty_slots += row['flight_empty_regions'] > 0
        self.away_decisions += row['away_member_decisions']
        self.away_reward += row['away_member_reward_sum']
        if row['flight_away_members']:
            self.away_system.add(row)

    def result(self):
        away = self.away_system.result()
        return dict(**super().result(), flight_region_crossings=self.crossings,
                    slots_with_away_members=self.away_slots, slots_with_empty_regions=self.empty_slots,
                    away_member_decisions=self.away_decisions,
                    away_member_reward_per_decision=self.away_reward / self.away_decisions if self.away_decisions else None,
                    away_slots_system_mean_delay_s=away['mean_delay_s'],
                    away_slots_system_failure_rate=away['failure_rate'] if self.away_slots else None)


def settings_from_snapshot(snapshot):
    settings = {name: cls(**snapshot[name]) for name, cls in (
        ('region', RegionConfig), ('user', UserConfig), ('channel', ChannelConfig),
        ('dag', DAGConfig), ('env', EnvConfig))}
    uav = dict(snapshot['uav'])
    uav['member_propulsion'] = MemberPropulsionConfig(**uav['member_propulsion'])
    settings['uav'] = UAVConfig(**uav)
    return settings
