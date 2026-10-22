"""Random 基线：随机飞行 → ERS 排序 → 随机放置 → 时延结算。"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
import uuid

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))

from env.contracts import PlacementDecision
from env.env import MRDTSEnv
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from methods.random.ers import ERS


class RandomStreams:
    """方法侧仅管理飞行与调度随机流；场景和 DAG 由环境按相同种子派生。"""

    def __init__(self, seed=42, episode=0):
        self.seed = seed
        self.episode = episode
        self.flight = np.random.default_rng(np.random.SeedSequence([seed, episode, 101]))
        self.scheduling = np.random.default_rng(np.random.SeedSequence([seed, episode, 102]))


def resolved_settings(users=None, dag_nodes=None):
    settings = dict(region=RegionConfig.default(), user=UserConfig.default(),
                    uav=UAVConfig.default(), channel=ChannelConfig.default(),
                    dag=DAGConfig.default(), env=EnvConfig.default())
    if users is not None:
        settings['user'] = replace(settings['user'], total_users=users)
    if dag_nodes is not None:
        settings['dag'] = replace(settings['dag'], n=dag_nodes)
    return settings


def build_scene(settings, streams):
    settings = {**settings, 'env': replace(settings['env'], env_seed=streams.seed)}
    environment = MRDTSEnv(settings)
    environment.generate_scene()
    environment.reset(epoch_index=streams.episode)
    return environment


def run_slot(environment, streams):
    # Master 固定；为每架 Member 独立采样两个水平动作分量。
    actions = streams.flight.uniform(-1., 1., (environment.members.member_uav_count, 2))
    begin = environment.begin_step(actions)
    runtime = environment._runtime
    plan = ERS(runtime).plan(begin.requests)
    candidates = {request.key: runtime.candidate_execution_nodes(
        request.owner_member, request.ground_device) for request in begin.requests}
    orders = {member.index: iter(order) for member, order in plan.member_orders.items()}
    ready = begin.ready_member_ids
    while ready:
        placements = {}
        for member_id in ready:
            key = next(orders[member_id])
            nodes = candidates[key.owner_member_id, key.user_id, key.dag_id]
            placements[key] = PlacementDecision(nodes[int(streams.scheduling.integers(len(nodes)))])
        ready = environment.submit_schedule(placements)
    return environment.end_step().slot_result, int(np.count_nonzero(begin.flight_violations))


def slot_metrics(result, episode, slot, flight_violations):
    count = len(result.dags)
    failed = sum(dag.failed for dag in result.dags)
    delays = [dag.completion_time - result.slot_start for dag in result.dags if not dag.failed]
    delay_sum = sum(delays)
    truncated_sum = delay_sum + failed * (result.slot_end - result.slot_start)
    return dict(episode=episode, slot=slot, dag_count=count, failed_dags=failed,
                failure_rate=failed / count if count else 0.,
                successful_delay_sum_s=delay_sum,
                successful_mean_delay_s=delay_sum / len(delays) if delays else None,
                truncated_delay_sum_s=truncated_sum,
                truncated_mean_delay_s=truncated_sum / count if count else None,
                flight_violations=flight_violations)


def summarize(rows):
    count = sum(row['dag_count'] for row in rows)
    failed = sum(row['failed_dags'] for row in rows)
    finished = count - failed
    return dict(slots_completed=len(rows), dag_count=count, failed_dags=failed,
                failure_rate=failed / count if count else 0.,
                successful_mean_delay_s=(sum(row['successful_delay_sum_s'] for row in rows) / finished
                                         if finished else None),
                truncated_mean_delay_s=(sum(row['truncated_delay_sum_s'] for row in rows) / count
                                        if count else None),
                flight_violations=sum(row['flight_violations'] for row in rows))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def git_output(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(*, seed=42, episodes=2, users=None, dag_nodes=None, output_root='results'):
    settings = resolved_settings(users, dag_nodes)
    settings['env'] = replace(settings['env'], env_seed=seed)
    steps_per_epoch = settings['env'].steps_per_epoch
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    output = output_root / 'random' / f'seed_{seed}' / stamp
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', dict(
        method='random', ordering='ers', seed=seed, episodes=episodes,
        environment={name: asdict(value) for name, value in settings.items()},
        delay_definition='completion_time - slot_start; unfinished DAGs capped at scheduling window',
    ))
    metadata = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                    method='random', seed=seed, branch=git_output('branch', '--show-current'),
                    commit=git_output('rev-parse', 'HEAD'), working_tree_status=git_output('status', '--short'),
                    python=platform.python_version(), numpy=np.__version__)
    write_json(output / 'metadata.json', metadata)
    rows = []
    try:
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as metrics_file, \
                (output / 'dags.csv').open('w', newline='', encoding='utf-8') as dags_file:
            metrics_writer = None
            dag_writer = csv.DictWriter(dags_file, fieldnames=[
                'episode', 'slot', 'owner_member_id', 'user_id', 'dag_id',
                'failed', 'completion_time_s', 'delay_s', 'truncated_delay_s',
            ])
            dag_writer.writeheader()
            for episode in range(episodes):
                streams = RandomStreams(seed, episode)
                # 同一空间随机流重建场景：区域、用户固定，UAV 回到初始位置。
                scene = build_scene(settings, streams)
                for slot in range(steps_per_epoch):
                    result, violations = run_slot(scene, streams)
                    row = slot_metrics(result, episode, slot, violations)
                    rows.append(row)
                    if metrics_writer is None:
                        metrics_writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                        metrics_writer.writeheader()
                    metrics_writer.writerow(row)
                    for dag in result.dags:
                        delay = None if dag.failed else dag.completion_time - result.slot_start
                        dag_writer.writerow(dict(
                            episode=episode, slot=slot, owner_member_id=dag.key[0], user_id=dag.key[1],
                            dag_id=dag.key[2], failed=dag.failed, completion_time_s=dag.completion_time,
                            delay_s=delay, truncated_delay_s=(result.slot_end - result.slot_start
                                                             if dag.failed else delay),
                        ))
                    metrics_file.flush()
                    dags_file.flush()
        write_json(output / 'summary.json', summarize(rows))
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        print(f'运行失败，已有结果保存在：{output}', file=sys.stderr)
        raise
    finally:
        metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
        metadata['slots_completed'] = len(rows)
        write_json(output / 'metadata.json', metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config' / 'methods' / 'random.toml')
    for name in ('seed', 'episodes', 'users', 'dag-nodes'):
        parser.add_argument(f'--{name}', type=int)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    with config_path.open('rb') as stream:
        config = tomllib.load(stream)['run']
    config.update({key: value for key, value in vars(args).items() if key != 'config' and value is not None})
    output = evaluate(**config)
    print(f'结果目录：{output}')
    print((output / 'summary.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
