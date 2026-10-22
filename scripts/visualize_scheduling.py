"""生成小规模单时隙事件回放，不修改框架或配置文件。

运行：uv run python scripts/visualize_scheduling.py
说明：scripts/scheduling_replay.md
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.randomness import RandomStreams
from env.communication.channel_model import ChannelModel
from env.contracts import DAGRequest, PlacementDecision
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User
from env.runtime.channel_queue import TransferStatus
from env.settings import UAVConfig, UserConfig
from env.simulator import Simulator
from env.types import EntityKind, EntityRef
from env.workload.dag_generator import DAG
from methods.random.ers import ERS
from methods.scheduling import run_online_schedule


def entity_id(entity):
    prefix = {EntityKind.GROUND_DEVICE: "g", EntityKind.MEMBER_UAV: "m", EntityKind.BS: "bs"}
    return "bs" if entity.kind is EntityKind.BS else f"{prefix[entity.kind]}{entity.index}"


def task_id(key):
    return f"d{key.dag_id}-t{key.node_id}"


def make_scene():
    region = Region(rng=RandomStreams().region)
    region.generate()
    count = region.config.region_count
    users = User(replace(UserConfig.default(), total_users=count, min_users_per_region=1,
                         area_fluctuation=0.0), rng=RandomStreams().user)
    users.generate_from_region(region)
    config = replace(UAVConfig.default(), master_uav_count=count,
                     member_uav_count=2 * count)
    masters = MasterUAV(config)
    masters.generate_from_region(region)
    members = MemberUAV(config)
    members.generate_from_region_and_user(region, users, masters)
    # 一用户时默认两架 Member 会重合；只在演示初始化中分散到附近区域网格。
    for user_index, region_id in enumerate(users.region_ids):
        indices = np.flatnonzero(members.region_ids[:members.bs_index] == region_id)
        cells = np.argwhere(region.region_map == region_id)
        xy = (cells[:, ::-1] + 0.5) * region.config.cell_size
        selected = []
        for offset in ((25.0, 15.0), (-85.0, 65.0)):
            distance = np.sum((xy - (users.positions[user_index, :2] + offset)) ** 2, axis=1)
            distance[selected] = np.inf
            selected.append(int(np.argmin(distance)))
        positions = np.column_stack((xy[selected], np.full(2, config.member_altitude)))
        members.update_positions(indices, positions)
    return region, users, members


def build_replay(load_scale=1.0, initial_region=1):
    """执行 begin_slot → method ERS → 在线决策/事件推进 → end_slot，提取真实回放记录。"""
    region, users, members = make_scene()
    environment = Simulator(members, ChannelModel(), users.config.transmit_power,
                              users.config.core_frequency)
    violations = environment.begin_slot(region, users.positions,
                                         np.zeros((members.member_uav_count, 2)))
    runtime = environment.runtime
    entities = []
    for entity, position in runtime.entity_positions.items():
        if entity.kind is EntityKind.GROUND_DEVICE:
            region_id = int(users.region_ids[entity.index])
            label = f"Ground {entity.index}"
        elif entity.kind is EntityKind.MEMBER_UAV:
            region_id = int(members.region_ids[entity.index])
            label = f"Member {entity.index}"
        else:
            region_id, label = 0, "BS"
        entities.append(dict(id=entity_id(entity), kind=entity.kind.value, label=label,
                             region=region_id, position=position.tolist(),
                             frequencies_hz=list(runtime.servers[entity].core_frequencies)))

    dag_definitions = []
    requests, execution_nodes = [], {}
    for user_index, region_id in enumerate(users.region_ids):
        owner_id = int(environment.user_member_ids[user_index])
        owner = EntityRef(EntityKind.MEMBER_UAV, owner_id)
        peer_id = next(int(i) for i in environment.member_ids_in_region(int(region_id)) if i != owner_id)
        ground = EntityRef(EntityKind.GROUND_DEVICE, user_index)
        locations = [owner, ground, EntityRef(EntityKind.MEMBER_UAV, peer_id), runtime.global_bs]
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        # 演示专用确定性负载：较长的计算段便于阅读；不覆盖 config/dag.toml。
        dag = DAG(4, edges, np.array([[120, 8e8], [70, 9e7], [100, 16e8], [80, 18e8]]) * load_scale,
                  {edge: size * load_scale for edge, size in zip(edges, (35, 60, 40, 55))})
        request = DAGRequest(user_index, dag, owner, ground)
        requests.append(request)
        execution_nodes.update({key: locations[key.node_id] for key in request.task_keys})
        dag_definitions.append(dict(id=f"d{user_index}", label=f"DAG {user_index}",
                                    region=int(region_id), ground=entity_id(ground), owner=entity_id(owner),
                                    peer=f"m{peer_id}", edges=[dict(source=f"d{user_index}-t{a}",
                                    target=f"d{user_index}-t{b}", kbit=dag.edge_features[(a, b)]) for a, b in edges]))

    plan = ERS(runtime).plan(requests)
    runtime.register_dags(requests)
    run_online_schedule(runtime, plan.member_orders, lambda key: PlacementDecision(execution_nodes[key]))
    result = environment.end_slot()
    # 排名仅用于展示 method 的序列，不参与环境的资源排序。
    ers_indices = {key: index for index, key in enumerate(plan.order)}
    request_by_key = {request.key: request for request in requests}
    tasks, transfers, events = [], [], []

    def event(at, kind, region_id, text, **extra):
        if at is not None:
            events.append(dict(at=float(at), kind=kind, region=region_id, text=text, **extra))

    for key, task in runtime.tasks.items():
        region_id = int(users.region_ids[key.user_id])
        tid = task_id(key)
        label = f"D{key.dag_id} / T{key.node_id}"
        execution = entity_id(task.execution_node) if task.execution_node is not None else None
        costs = plan.costs[key.owner_member_id, key.user_id, key.dag_id]
        tasks.append(dict(id=tid, label=label, node=key.node_id, dag=f"d{key.dag_id}", region=region_id,
                          execution=execution,
                          ers=ers_indices[key], decision_at=task.decision_at,
                          rank_s=plan.ranks[key], average_compute_s=costs.average_compute_s[key.node_id],
                          average_edge_comm_s={str(child): value for (parent, child), value
                                               in costs.average_edge_comm_s.items() if parent == key.node_id},
                          input_kbit=task.input_bits / 1000.0, cycles=task.cpu_cycles,
                          predecessors=[task_id(k) for k in sorted(task.predecessors)],
                          input_arrival=task.input_arrival_at, ready=task.data_ready_at,
                          queued=task.compute_queue_enter_at, start=task.compute_start_at,
                          finish=task.compute_finish_at, core=task.execution_core_id,
                          failed_at=task.failed_at, final_status=task.status.value,
                          compute_energy_j=task.compute_energy_j))
        event(task.decision_at, "placement", region_id, f"{label} 确定卸载目标", task=tid)
        event(task.input_arrival_at, "input_arrival", region_id, f"{label} 原始输入到达", task=tid)
        for parent, at in sorted(task.predecessor_arrival_at.items()):
            event(at, "result_arrival", region_id, f"T{parent.node_id} → T{key.node_id} 中间结果到达", task=tid)
        event(task.data_ready_at, "data_ready", region_id, f"{label} 数据全部就绪，进入计算 FIFO", task=tid)
        event(task.compute_start_at, "compute_start", region_id,
              f"{label} 开始计算 · {execution} / C{task.execution_core_id}", task=tid)
        event(task.compute_finish_at, "compute_finish", region_id, f"{label} 计算完成", task=tid)
        event(task.failed_at, "task_failed", region_id, f"{label} 截止未完成，任务失败", task=tid)
        records = [(None, task.input_record), *sorted(task.predecessor_records.items())]
        request = request_by_key[key.owner_member_id, key.user_id, key.dag_id]
        for parent, record in records:
            if record is None:
                continue  # 未调度任务尚无传输记录。
            route = (runtime.route_planner.input_route(request.ground_device, request.owner_member, task.execution_node)
                     if parent is None else runtime.route_planner.predecessor_route(
                         runtime.trace(parent).execution_node, task.execution_node,
                         request.ground_device, request.owner_member))
            for hop, (channel, transfer_id) in enumerate(zip(route.hops, record.transfer_ids)):
                job = runtime.transfer_jobs[transfer_id]
                completed = job.status is TransferStatus.FINISHED
                finish = job.finish_at if completed else None
                display_end = min(job.finish_at, result.slot_end) if job.start_at is not None else None
                name = f"T{key.node_id} 输入" if parent is None else f"T{parent.node_id} → T{key.node_id} 结果"
                source, target = entity_id(channel.source), entity_id(channel.target)
                energy = ((display_end - job.start_at) * runtime.transmit_powers[channel.source]
                          if display_end is not None and channel.source.kind is EntityKind.MEMBER_UAV else 0.0)
                transfers.append(dict(id=transfer_id, task=tid, region=region_id,
                                      kind="input" if parent is None else "result", label=name,
                                      predecessor=task_id(parent) if parent is not None else None,
                                      hop=hop, hop_count=len(record.transfer_ids), source=source, target=target,
                                      queue_order=None, ers=ers_indices[key], source_ready=job.source_ready_at,
                                      duration=job.duration_s, start=job.start_at, finish=finish,
                                      display_end=display_end, completed=completed, energy_j=energy))
                event(job.start_at, "transfer_start", region_id,
                      f"{name} · {source} → {target} 开始传输（第 {hop + 1} 跳）", task=tid, transfer=transfer_id)
                event(finish, "transfer_finish", region_id,
                      f"{name} · {source} → {target} 传输完成", task=tid, transfer=transfer_id)

    channel_jobs = {}
    for transfer in transfers:
        channel_jobs.setdefault((transfer['source'], transfer['target']), []).append(transfer)
    for jobs in channel_jobs.values():
        jobs.sort(key=lambda job: (job['start'] if job['start'] is not None else float('inf'),
                                   job['source_ready'] if job['source_ready'] is not None else float('inf'),
                                   int(job['id'].split('-')[1])))
        for index, job in enumerate(jobs):
            job['queue_order'] = index

    for dag, outcome in zip(dag_definitions, sorted(result.dags, key=lambda item: item.key[1])):
        dag.update(completion=outcome.completion_time, failed=outcome.failed)
        event(0.0, "dag_submit", dag["region"], f"{dag['label']} 登记：各子任务等待在线卸载决策")
        event(outcome.completion_time, "dag_finish", dag["region"], f"{dag['label']} 全部子任务完成")
    event(result.slot_end, "slot_close", 0, "时隙截止：未完成 DAG 失败，运行时关闭并释放")
    # 同时刻事件作为一批展示；顺序仅用于可读性，不宣称是内核微观处理顺序。
    priority = {"dag_submit": 0, "transfer_finish": 1, "compute_finish": 1, "placement": 2,
                "input_arrival": 3, "result_arrival": 3, "data_ready": 4,
                "compute_start": 5, "transfer_start": 5, "dag_finish": 6, "slot_close": 9}
    events.sort(key=lambda e: (e["at"], priority.get(e["kind"], 8), e["region"], e["text"]))
    return dict(meta=dict(slot_start=result.slot_start, deadline=result.slot_end,
                          initial_region=initial_region, region_count=region.config.region_count,
                          load_scale=load_scale, flight_duration=members.env_config.flight_duration_s,
                          flight_violations=violations.tolist()),
                map=dict(side=region.config.side_length, cells=region.region_map.tolist()),
                entities=entities, dags=dag_definitions, tasks=tasks, transfers=transfers, events=events,
                summary=dict(finished_dags=len(result.finished_dags), failed_dags=len(result.failed_dags),
                             compute_energy_j=result.compute_energy_j, tx_energy_j=result.tx_energy_j))


def write_replay(data, output):
    output = Path(output).resolve()
    template = Path(__file__).with_name("templates") / "scheduling_replay.html"
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    html = template.read_text(encoding="utf-8").replace("__REPLAY_DATA__", payload.replace("<", "\\u003c"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    output.with_suffix(".json").write_text(payload, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=int, default=1, help="页面初始选中区域，默认 1")
    parser.add_argument("--load-scale", type=float, default=1.0, help="演示输入、结果和计算量倍率，默认 1")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output") / "scheduling_replay.html")
    args = parser.parse_args()
    try:
        data = build_replay(args.load_scale, args.region)
        output = write_replay(data, args.output)
    except ValueError as error:
        parser.error(str(error))
    print(f"HTML: {output}")
    print(f"JSON: {output.with_suffix('.json')}")
    print(f"DAGs: {data['summary']['finished_dags']} finished, {data['summary']['failed_dags']} failed")


if __name__ == "__main__":
    main()
