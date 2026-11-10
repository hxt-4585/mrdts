"""Configuration, portable environment snapshots and atomic experiment files."""
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import tomllib

import torch

from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.settings.uav_config import MemberPropulsionConfig

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    with Path(path or ROOT / 'config' / 'methods' / 'mrdts_delay.toml').open('rb') as stream:
        return tomllib.load(stream)


def pack_settings(settings):
    return {name: asdict(value) for name, value in settings.items()}


def unpack_settings(snapshot):
    classes = dict(env=EnvConfig, region=RegionConfig, uav=UAVConfig,
                   user=UserConfig, channel=ChannelConfig, dag=DAGConfig)
    restored = {}
    for name, cls in classes.items():
        fields = dict(snapshot[name])
        if name == 'uav':
            fields['member_propulsion'] = MemberPropulsionConfig(**fields['member_propulsion'])
        restored[name] = cls(**fields)
    return restored


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    torch.save(state, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def append_rows(path, rows):
    path = Path(path)
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    dags = sum(row['dag_count'] for row in rows)
    delay = float(sum(row['delay_sum_s'] for row in rows))
    failed = int(sum(row['failed_dags'] for row in rows))
    return dict(slots=len(rows), dag_count=dags, delay_sum_s=delay, mean_delay_s=delay / dags,
                failed_dags=failed, failure_rate=failed / dags,
                flight_violations=sum(row['flight_violations'] for row in rows))
