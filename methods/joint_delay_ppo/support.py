"""Environment snapshots and atomic local experiment records."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
import tomllib

from env.env import MRDTSEnv
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.settings.uav_config import MemberPropulsionConfig

ROOT = Path(__file__).resolve().parents[2]
CONSTRUCTORS = dict(env=EnvConfig, region=RegionConfig, uav=UAVConfig,
                    user=UserConfig, channel=ChannelConfig, dag=DAGConfig)


def load_config(path=None):
    with Path(path or ROOT / 'config/methods/joint_delay_ppo.toml').open('rb') as stream:
        return tomllib.load(stream)


def environment_snapshot(config):
    settings = {name: constructor.default() for name, constructor in CONSTRUCTORS.items()}
    train = config['train']
    settings['env'] = replace(settings['env'], steps_per_epoch=train['slots'])
    if train.get('users') is not None:
        settings['user'] = replace(settings['user'], total_users=train['users'])
    if train.get('dag_nodes') is not None:
        settings['dag'] = replace(settings['dag'], n=train['dag_nodes'])
    return {name: asdict(value) for name, value in settings.items()}


def make_environment(snapshot):
    settings = {}
    for name, constructor in CONSTRUCTORS.items():
        values = dict(snapshot[name])
        if name == 'uav':
            values['member_propulsion'] = MemberPropulsionConfig(**values['member_propulsion'])
        settings[name] = constructor(**values)
    env = MRDTSEnv(settings)
    env.generate_scene()
    env.reset(epoch_index=0)
    return env


def replace_file(source, target):
    """Allow transient Windows reader handles to close before atomic replacement."""
    delay = .02
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(delay)
            delay = min(delay * 2., .5)


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    replace_file(pending, path)


def cpu_weights(model):
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def aggregate(rows):
    delay = sum(row['delay_sum_s'] for row in rows)
    count = sum(row['dag_count'] for row in rows)
    failed = sum(row['failed_dags'] for row in rows)
    return dict(slots=len(rows), dag_count=count, delay_sum_s=delay, mean_delay_s=delay / count,
                failed_dags=failed, failure_rate=failed / count,
                flight_violations=sum(row['flight_violations'] for row in rows),
                member_decisions=sum(row['member_decisions'] for row in rows))

