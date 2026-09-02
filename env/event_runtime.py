"""单个时隙内多 DAG 传输与计算的离散事件运行时。"""

from collections import defaultdict
from dataclasses import dataclass
import heapq
from itertools import count
from typing import Mapping

import numpy as np

from env.channel_model import ChannelModel, LinkType
from env.channel_queue import DirectedChannelKey, DirectedChannelState, EntityKind, EntityRef, TransferJob
from env.dag_generator import DAG
from env.dag_runtime import DAGRuntime
from env.routing import RoutePlanner
from env.server_queue import ServerState
from env.task_runtime import TaskKey, TaskRuntime, TransferRecord
from methods.contracts import PlacementDecision


@dataclass(frozen=True)
class ServerSpec:
    core_frequencies: tuple[float, ...]
    capacitance_factor: float


@dataclass
class _TransferBinding:
    task_key: TaskKey
    record: TransferRecord
    power_w: float
    next_transfer_id: str | None = None
    predecessor: TaskKey | None = None
    is_input: bool = False


class SchedulingRuntime:
    """将 placement 解释为传输、数据依赖及 FIFO 计算事件。"""

    def __init__(
        self,
        channel_model: ChannelModel,
        entity_positions: Mapping[EntityRef, np.ndarray],
        servers: Mapping[EntityRef, ServerSpec],
        transmit_powers: Mapping[EntityRef, float],
        directed_bandwidth_hz: Mapping[DirectedChannelKey, float] | None = None,
        member_regions: Mapping[EntityRef, int] | None = None,
        ground_owner_members: Mapping[EntityRef, EntityRef] | None = None,
    ):
        self.channel_model = channel_model
        self.entity_positions = {
            entity: np.asarray(position, dtype=float) for entity, position in entity_positions.items()
        }
        ground_entities = {
            entity for entity in self.entity_positions if entity.kind is EntityKind.GROUND_DEVICE
        }
        ground_servers = {entity for entity in servers if entity.kind is EntityKind.GROUND_DEVICE}
        if ground_servers != ground_entities:
            raise ValueError("每个地面设备必须恰有一个本地计算服务器")
        for ground in ground_servers:
            spec = servers[ground]
            if len(spec.core_frequencies) != 1 or spec.capacitance_factor != 0.0:
                raise ValueError("地面设备服务器必须为单核且采用零计算能耗")
        bs_entities = {entity for entity in servers if entity.kind is EntityKind.BS}
        if len(bs_entities) != 1:
            raise ValueError("调度运行时必须恰有一个全局 BS")
        self.global_bs = next(iter(bs_entities))
        self.servers = {
            entity: ServerState(entity, tuple(spec.core_frequencies), spec.capacitance_factor)
            for entity, spec in servers.items()
        }
        self.transmit_powers = {entity: float(power) for entity, power in transmit_powers.items()}
        self.directed_bandwidth_hz = dict(directed_bandwidth_hz or {})
        self.member_regions: dict[EntityRef, int] | None = None
        self.ground_owner_members: dict[EntityRef, EntityRef] | None = None
        self.route_planner = RoutePlanner()
        self.channels: dict[DirectedChannelKey, DirectedChannelState] = {}
        self.tasks: dict[TaskKey, TaskRuntime] = {}
        self.dag_runtimes: dict[tuple[int, int, int], DAGRuntime] = {}
        self._transfer_channel: dict[str, DirectedChannelKey] = {}
        self._transfer_bindings: dict[str, _TransferBinding] = {}
        self._outgoing_transfers: dict[TaskKey, list[str]] = defaultdict(list)
        self._local_successors: dict[TaskKey, list[TaskKey]] = defaultdict(list)
        self._ready_for_server: set[TaskKey] = set()
        self._events: list[tuple[float, int, str, object]] = []
        self._event_counter = count()
        self._transfer_counter = count()
        self._next_ers_seq = 0
        self.now = 0.0
        if member_regions is not None or ground_owner_members is not None:
            if member_regions is None or ground_owner_members is None:
                raise ValueError("member_regions 与 ground_owner_members 必须同时提供")
            self.update_topology(member_regions, ground_owner_members)

    def submit_dag(
        self,
        dag_id: int,
        dag: DAG,
        owner_member: EntityRef,
        ground_device: EntityRef,
        placements: Mapping[int, PlacementDecision],
        epoch_start: float,
    ) -> tuple[TaskKey, ...]:
        if epoch_start < self.now:
            raise ValueError("DAG 提交时刻不能早于当前运行时刻")
        if owner_member.kind is not EntityKind.MEMBER_UAV:
            raise ValueError("owner_member 必须是 Member UAV")
        if ground_device.kind is not EntityKind.GROUND_DEVICE:
            raise ValueError("ground_device 必须是地面设备")
        if set(placements) != set(range(dag.node_num)):
            raise ValueError("每个子任务必须恰有一个 placement")
        if (owner_member.index, ground_device.index, dag_id) in self.dag_runtimes:
            raise ValueError("同一 owner/user/dag_id 的 DAG 已提交")
        self._validate_submission_topology(owner_member, ground_device, placements)

        self.now = float(epoch_start)
        ordered_nodes = sorted(placements, key=lambda node: (placements[node].ers_seq, node))
        global_ers = {node: self._next_ers_seq + index for index, node in enumerate(ordered_nodes)}
        self._next_ers_seq += dag.node_num
        predecessors = {node: set() for node in range(dag.node_num)}
        for parent, child in dag.edges:
            predecessors[child].add(parent)
        keys = tuple(TaskKey(owner_member.index, ground_device.index, dag_id, node) for node in range(dag.node_num))
        node_keys = {key.node_id: key for key in keys}
        for node, key in node_keys.items():
            decision = placements[node]
            if decision.execution_node not in self.servers:
                raise ValueError(f"执行节点 {decision.execution_node} 没有计算服务器")
            self.tasks[key] = TaskRuntime(
                key=key,
                execution_node=decision.execution_node,
                ers_seq=global_ers[node],
                cpu_cycles=float(dag.node_features[node, 1]),
                input_bits=float(dag.node_features[node, 0]) * 8.0 * 1024.0,
                predecessors={node_keys[parent] for parent in predecessors[node]},
            )
        self.dag_runtimes[(owner_member.index, ground_device.index, dag_id)] = DAGRuntime(dag)

        pending: list[tuple[DirectedChannelKey, TransferJob, _TransferBinding]] = []
        for node, key in node_keys.items():
            task = self.tasks[key]
            route = self.route_planner.input_route(ground_device, owner_member, task.execution_node)
            if not route.hops:
                task.input_record = TransferRecord(
                    (), start_at=float(epoch_start), finish_at=float(epoch_start)
                )
                if task.mark_input_arrived(epoch_start):
                    self._ready_for_server.add(task.key)
            else:
                task.input_record = self._append_route(
                    pending, route.hops, task, task.input_bits, epoch_start, is_input=True
                )
        for parent, child in dag.edges:
            parent_task = self.tasks[node_keys[parent]]
            child_task = self.tasks[node_keys[child]]
            result_bits = float(dag.edge_features[(parent, child)]) * 8.0 * 1024.0
            route = self.route_planner.predecessor_route(
                parent_task.execution_node,
                child_task.execution_node,
                ground_device,
                owner_member,
            )
            if not route.hops:
                record = TransferRecord(())
                child_task.predecessor_records[parent_task.key] = record
                self._local_successors[parent_task.key].append(child_task.key)
            else:
                record = self._append_route(
                    pending, route.hops, child_task, result_bits, None, predecessor=parent_task.key
                )
                child_task.predecessor_records[parent_task.key] = record
                self._outgoing_transfers[parent_task.key].append(record.transfer_ids[0])

        for key, job, binding in sorted(pending, key=lambda item: (item[0].source, item[0].target, item[1].ers_seq, item[1].transfer_id)):
            state = self.channels.setdefault(key, DirectedChannelState(key))
            state.enqueue(job)
            self._transfer_channel[job.transfer_id] = key
            self._transfer_bindings[job.transfer_id] = binding
        self._dispatch(self.now)
        return keys

    def advance_until(self, absolute_time: float) -> None:
        absolute_time = float(absolute_time)
        if absolute_time < self.now:
            raise ValueError("不能将运行时倒退到过去")
        while self._events and self._events[0][0] <= absolute_time:
            timestamp = self._events[0][0]
            batch: list[tuple[float, int, str, object]] = []
            while self._events and self._events[0][0] == timestamp:
                batch.append(heapq.heappop(self._events))
            self.now = timestamp
            for _, _, event_type, payload in sorted(batch, key=lambda event: event[2]):
                if event_type == "compute_finish":
                    self._handle_compute_finish(payload, timestamp)
                else:
                    self._handle_transfer_finish(payload, timestamp)
            self._dispatch(timestamp)
        self.now = absolute_time

    def trace(self, task_key: TaskKey) -> TaskRuntime:
        return self.tasks[task_key]

    def channel_state(self, key: DirectedChannelKey) -> DirectedChannelState:
        return self.channels[key]

    def update_entity_positions(self, positions: Mapping[EntityRef, np.ndarray]) -> None:
        """在新时隙提交前更新实体快照，不回溯修改既有传输。"""
        for entity, position in positions.items():
            if entity not in self.entity_positions:
                raise KeyError(f"运行时不认识实体 {entity}")
            position = np.asarray(position, dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("实体位置必须是包含有限数值的 shape=(3,) 向量")
            self.entity_positions[entity] = position

    def update_topology(
        self,
        member_regions: Mapping[EntityRef, int],
        ground_owner_members: Mapping[EntityRef, EntityRef],
    ) -> None:
        """刷新当前时隙内的 Member 区域和用户关联。"""
        expected_members = {
            entity for entity in self.servers if entity.kind is EntityKind.MEMBER_UAV
        }
        supplied_members = set(member_regions)
        if supplied_members != expected_members:
            raise ValueError("拓扑必须为全部 Member UAV 提供区域编号")
        normalized_regions: dict[EntityRef, int] = {}
        for member, region_id in member_regions.items():
            if member.kind is not EntityKind.MEMBER_UAV:
                raise ValueError("member_regions 的键必须是 Member UAV")
            if isinstance(region_id, bool) or int(region_id) != region_id or int(region_id) <= 0:
                raise ValueError("Member UAV 的区域编号必须为正整数")
            normalized_regions[member] = int(region_id)

        expected_grounds = {
            entity for entity in self.entity_positions if entity.kind is EntityKind.GROUND_DEVICE
        }
        if set(ground_owner_members) != expected_grounds:
            raise ValueError("拓扑必须为全部地面设备提供关联 Member UAV")
        normalized_associations: dict[EntityRef, EntityRef] = {}
        for ground, owner in ground_owner_members.items():
            if ground.kind is not EntityKind.GROUND_DEVICE:
                raise ValueError("ground_owner_members 的键必须是地面设备")
            if owner not in normalized_regions:
                raise ValueError("地面设备必须关联到已登记的 Member UAV")
            normalized_associations[ground] = owner
        self.member_regions = normalized_regions
        self.ground_owner_members = normalized_associations

    def _append_route(
        self,
        pending: list[tuple[DirectedChannelKey, TransferJob, _TransferBinding]],
        hops: tuple[DirectedChannelKey, ...],
        task: TaskRuntime,
        payload_bits: float,
        source_ready_at: float | None,
        is_input: bool = False,
        predecessor: TaskKey | None = None,
    ) -> TransferRecord:
        transfer_ids = tuple(f"tx-{next(self._transfer_counter)}" for _ in hops)
        record = TransferRecord(transfer_ids)
        for index, hop in enumerate(hops):
            transfer_id = transfer_ids[index]
            job = TransferJob(
                transfer_id=transfer_id,
                ers_seq=task.ers_seq,
                source_ready_at=source_ready_at if index == 0 else None,
                duration_s=self._duration_s(hop, payload_bits),
            )
            pending.append(
                (
                    hop,
                    job,
                    _TransferBinding(
                        task.key,
                        record,
                        self.transmit_powers[hop.source],
                        next_transfer_id=transfer_ids[index + 1] if index + 1 < len(hops) else None,
                        predecessor=predecessor,
                        is_input=is_input,
                    ),
                )
            )
        return record

    def _duration_s(self, hop: DirectedChannelKey, payload_bits: float) -> float:
        transmitter = self.entity_positions[hop.source]
        receiver = self.entity_positions[hop.target]
        power = self.transmit_powers[hop.source]
        bandwidth_hz = self.directed_bandwidth_hz.get(hop, self._default_bandwidth_hz(hop))
        metrics = self.channel_model.calculate_link(
            transmitter, receiver, power, self._link_type(hop), bandwidth_hz
        )
        return float(payload_bits) / metrics.rate_bps

    def _validate_submission_topology(
        self,
        owner_member: EntityRef,
        ground_device: EntityRef,
        placements: Mapping[int, PlacementDecision],
    ) -> None:
        if self.member_regions is None or self.ground_owner_members is None:
            raise RuntimeError("提交 DAG 前必须提供当前时隙拓扑")
        if self.ground_owner_members[ground_device] != owner_member:
            raise ValueError("地面设备当前关联的 Member UAV 与 owner_member 不一致")
        owner_region_id = self.member_regions[owner_member]
        for decision in placements.values():
            execution_node = decision.execution_node
            if execution_node.kind is EntityKind.GROUND_DEVICE:
                if execution_node != ground_device:
                    raise ValueError("子任务只能在自身地面设备本地执行")
            elif execution_node.kind is EntityKind.MEMBER_UAV:
                if self.member_regions.get(execution_node) != owner_region_id:
                    raise ValueError("Member UAV 执行节点必须与任务 owner 位于同一区域")
            elif execution_node.kind is EntityKind.BS:
                if execution_node != self.global_bs:
                    raise ValueError("子任务只能卸载到唯一的全局 BS")
            else:
                raise ValueError("执行节点只能是自身地面设备、Member UAV 或 BS")

    def _default_bandwidth_hz(self, hop: DirectedChannelKey) -> float:
        config = self.channel_model.config
        if hop.source.kind is EntityKind.GROUND_DEVICE and hop.target.kind is EntityKind.MEMBER_UAV:
            return config.ground_to_air_bandwidth_mhz * 1e6
        if hop.source.kind is EntityKind.MEMBER_UAV and hop.target.kind is EntityKind.MEMBER_UAV:
            return config.air_to_air_bandwidth_mhz * 1e6
        if hop.source.kind is EntityKind.MEMBER_UAV and hop.target.kind in {EntityKind.BS, EntityKind.GROUND_DEVICE}:
            return config.air_to_ground_bandwidth_mhz * 1e6
        if hop.source.kind is EntityKind.BS and hop.target.kind is EntityKind.MEMBER_UAV:
            return config.ground_to_air_bandwidth_mhz * 1e6
        raise ValueError(f"没有 {hop} 的默认带宽")

    @staticmethod
    def _link_type(hop: DirectedChannelKey) -> LinkType:
        if hop.source.kind is EntityKind.MEMBER_UAV and hop.target.kind is EntityKind.MEMBER_UAV:
            return LinkType.AIR_TO_AIR
        if hop.source.kind is EntityKind.MEMBER_UAV:
            return LinkType.AIR_TO_GROUND
        return LinkType.GROUND_TO_AIR

    def _handle_transfer_finish(self, transfer_id: str, at: float) -> None:
        state = self.channels[self._transfer_channel[transfer_id]]
        job = state.finish_active(at)
        if job.transfer_id != transfer_id:
            raise RuntimeError("传输完成事件与活动信道队首不一致")
        binding = self._transfer_bindings[transfer_id]
        if binding.next_transfer_id is not None:
            next_key = self._transfer_channel[binding.next_transfer_id]
            self.channels[next_key].mark_source_ready(binding.next_transfer_id, at)
            return
        binding.record.finish_at = at
        if binding.predecessor is not None:
            if self.tasks[binding.task_key].mark_predecessor_arrived(binding.predecessor, at):
                self._ready_for_server.add(binding.task_key)
        elif binding.is_input and self.tasks[binding.task_key].mark_input_arrived(at):
            self._ready_for_server.add(binding.task_key)

    def _handle_compute_finish(self, payload: tuple[EntityRef, int], at: float) -> None:
        server_node, core_id = payload
        record = self.servers[server_node].finish(core_id, at)
        task = self.tasks[record.task_key]
        task.mark_compute_finished(at)
        dag_key = (task.key.owner_member_id, task.key.user_id, task.key.dag_id)
        self.dag_runtimes[dag_key].mark_finished(task.key.node_id, at)
        for transfer_id in self._outgoing_transfers[task.key]:
            key = self._transfer_channel[transfer_id]
            self.channels[key].mark_source_ready(transfer_id, at)
        for child_key in self._local_successors[task.key]:
            child = self.tasks[child_key]
            child.predecessor_records[task.key].start_at = at
            child.predecessor_records[task.key].finish_at = at
            if child.mark_predecessor_arrived(task.key, at):
                self._ready_for_server.add(child_key)

    def _dispatch(self, at: float) -> None:
        for task_key in sorted(
            self._ready_for_server,
            key=lambda key: (self.tasks[key].data_ready_at, self.tasks[key].ers_seq, key),
        ):
            task = self.tasks[task_key]
            self.servers[task.execution_node].enqueue(
                task.key, task.cpu_cycles, task.data_ready_at, task.ers_seq
            )
        self._ready_for_server.clear()
        for key in sorted(self.channels, key=lambda channel: (channel.source, channel.target)):
            state = self.channels[key]
            job = state.start_head(at)
            if job is None:
                continue
            binding = self._transfer_bindings[job.transfer_id]
            if binding.record.start_at is None:
                binding.record.start_at = at
            binding.record.tx_energy_j += binding.power_w * job.duration_s
            heapq.heappush(
                self._events, (job.finish_at, next(self._event_counter), "transfer_finish", job.transfer_id)
            )
        for server in self.servers.values():
            for record in server.dispatch(at):
                task = self.tasks[record.task_key]
                task.mark_compute_started(record.core_id, at, record.energy_j)
                heapq.heappush(
                    self._events,
                    (record.finish_at, next(self._event_counter), "compute_finish", (server.node, record.core_id)),
                )
