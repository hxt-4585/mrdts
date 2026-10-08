"""单个时隙内多 DAG 传输与计算的离散事件运行时。"""

from collections import defaultdict
from copy import copy
from dataclasses import dataclass
import heapq
from itertools import count
from typing import Iterable, Mapping

import numpy as np

from env.communication.channel_model import ChannelModel, LinkType
from env.communication.routing import RoutePlanner
from env.contracts import DAGRequest, PlacementDecision
from env.runtime.channel_queue import DirectedChannelState, TransferJob
from env.runtime.dag_runtime import DAGRuntime
from env.runtime.server_queue import ServerState
from env.runtime.slot_result import DAGResult, SlotResult
from env.runtime.task_runtime import TaskRuntime, TaskStatus, TransferRecord
from env.types import DirectedChannelKey, EntityKind, EntityRef, TaskKey
from env.workload.dag_generator import DAG


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


@dataclass
class _PreparedDAG:
    key: tuple[int, int, int]
    tasks: dict[TaskKey, TaskRuntime]
    runtime: DAGRuntime
    pending: list[tuple[DirectedChannelKey, TransferJob, _TransferBinding]]
    outgoing: dict[TaskKey, list[str]]
    local_successors: dict[TaskKey, list[TaskKey]]


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
        *,
        slot_start: float = 0.0,
        max_duration_s: float = 1.0,
    ):
        self._slot_start = float(slot_start)
        duration = float(max_duration_s)
        self._deadline = self._slot_start + duration
        self._closed = False
        self._result: SlotResult | None = None
        self.channel_model = channel_model
        self.entity_positions = {
            entity: np.array(position, dtype=float, copy=True) for entity, position in entity_positions.items()
        }
        bs_entities = {entity for entity in servers if entity.kind is EntityKind.BS}
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
        self._next_priority_seq = 0
        self.now = self.slot_start
        if member_regions is not None or ground_owner_members is not None:
            self.update_topology(member_regions, ground_owner_members)

    @property
    def slot_start(self) -> float:
        return self._slot_start

    @property
    def deadline(self) -> float:
        return self._deadline

    @property
    def closed(self) -> bool:
        return self._closed

    def submit_dag(
        self,
        dag_id: int,
        dag: DAG,
        owner_member: EntityRef,
        ground_device: EntityRef,
        placements: Mapping[int, PlacementDecision],
        epoch_start: float,
    ) -> tuple[TaskKey, ...]:
        """单 DAG 便捷入口；统一多 DAG 排序请使用 submit_dags。"""
        request = DAGRequest(dag_id, dag, owner_member, ground_device)
        return self.submit_dags(
            [request], {key: placements[key.node_id] for key in request.task_keys}, epoch_start)

    def submit_dags(
        self,
        requests: Iterable[DAGRequest],
        placements: Mapping[TaskKey, PlacementDecision],
        epoch_start: float,
    ) -> tuple[TaskKey, ...]:
        """整批准备成功后统一入队和启动，保留跨 DAG 的相对优先顺序。

        本批序号压缩为连续编号并追加在已提交批次之后；已经启动的任务
        不参与重新排序。一个 Member 的同批 DAG 应先一起调用排序组件。
        """
        epoch_start = float(epoch_start)
        requests = tuple(requests)
        order = sorted(placements, key=lambda key: placements[key].priority_seq)
        global_priority = {key: self._next_priority_seq + i for i, key in enumerate(order)}
        # 预备阶段仅修改局部状态，连传输 ID 计数也在成功后才提交。
        transfer_counter = copy(self._transfer_counter)
        prepared = [self._prepare_dag(request, placements, global_priority, epoch_start, transfer_counter)
                    for request in sorted(requests, key=lambda item: item.key)]
        pending = [item for dag in prepared for item in dag.pending]
        for dag in prepared:
            self.tasks.update(dag.tasks)
            self.dag_runtimes[dag.key] = dag.runtime
            for key, transfers in dag.outgoing.items():
                self._outgoing_transfers[key].extend(transfers)
            for key, children in dag.local_successors.items():
                self._local_successors[key].extend(children)
            self._ready_for_server.update(key for key, task in dag.tasks.items() if task.data_ready_at is not None)
        for key, job, binding in sorted(pending, key=lambda item: (
            item[0].source, item[0].target, item[1].priority_seq, item[1].transfer_id
        )):
            state = self.channels.setdefault(key, DirectedChannelState(key))
            state.enqueue(job)
            self._transfer_channel[job.transfer_id] = key
            self._transfer_bindings[job.transfer_id] = binding
        self._next_priority_seq += len(order)
        self._transfer_counter = transfer_counter
        self._dispatch(self.now)
        return tuple(key for dag in prepared for key in dag.tasks)

    def _prepare_dag(
        self, request: DAGRequest, placements: Mapping[TaskKey, PlacementDecision],
        global_priority: Mapping[TaskKey, int], epoch_start: float, transfer_counter: count,
    ) -> _PreparedDAG:
        dag, ground_device, owner_member = request.dag, request.ground_device, request.owner_member
        prepared = _PreparedDAG(request.key, {}, DAGRuntime(dag), [], defaultdict(list), defaultdict(list))
        predecessors = {node: set() for node in range(dag.node_num)}
        for parent, child in dag.edges:
            predecessors[child].add(parent)
        node_keys = {key.node_id: key for key in request.task_keys}
        for node, key in node_keys.items():
            decision = placements[key]
            prepared.tasks[key] = TaskRuntime(
                key=key,
                execution_node=decision.execution_node,
                priority_seq=global_priority[key],
                cpu_cycles=float(dag.node_features[node, 1]),
                input_bits=float(dag.node_features[node, 0]) * 1000.0,
                predecessors={node_keys[parent] for parent in predecessors[node]},
            )
        for node, key in node_keys.items():
            task = prepared.tasks[key]
            route = self.route_planner.input_route(ground_device, owner_member, task.execution_node)
            if not route.hops or task.input_bits == 0:
                task.input_record = TransferRecord(
                    (), start_at=float(epoch_start), finish_at=float(epoch_start)
                )
                task.mark_input_arrived(epoch_start)
            else:
                task.input_record = self._append_route(
                    prepared.pending, route.hops, task, task.input_bits, epoch_start,
                    transfer_counter, is_input=True
                )
        for parent, child in dag.edges:
            parent_task = prepared.tasks[node_keys[parent]]
            child_task = prepared.tasks[node_keys[child]]
            result_bits = float(dag.edge_features[(parent, child)]) * 1000.0
            route = self.route_planner.predecessor_route(
                parent_task.execution_node,
                child_task.execution_node,
                ground_device,
                owner_member,
            )
            if not route.hops or result_bits == 0:
                record = TransferRecord(())
                child_task.predecessor_records[parent_task.key] = record
                prepared.local_successors[parent_task.key].append(child_task.key)
            else:
                record = self._append_route(
                    prepared.pending, route.hops, child_task, result_bits, None,
                    transfer_counter, predecessor=parent_task.key
                )
                child_task.predecessor_records[parent_task.key] = record
                prepared.outgoing[parent_task.key].append(record.transfer_ids[0])
        return prepared

    def advance_until(self, absolute_time: float) -> None:
        absolute_time = float(absolute_time)
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
        if self.now == self.deadline:
            self._close_slot()

    def finish_slot(self) -> SlotResult:
        """执行到截止时刻并关闭；重复调用只返回同一份结果。"""
        if not self.closed:
            self.advance_until(self.deadline)
        return self._result

    def _close_slot(self) -> None:
        # 启动时预记了完整作业能耗；取消时扣除窗口以后的部分。
        for channel in self.channels.values():
            job = channel.active
            if job is not None:
                binding = self._transfer_bindings[job.transfer_id]
                binding.record.tx_energy_j -= binding.power_w * (job.finish_at - self.deadline)
                binding.record.tx_energy_j = max(0.0, binding.record.tx_energy_j)
            channel.queued.clear()
            channel.active = None
        for server in self.servers.values():
            for record in server.running.values():
                elapsed = self.deadline - record.start_at
                duration = record.finish_at - record.start_at
                self.tasks[record.task_key].compute_energy_j = record.energy_j * elapsed / duration
            server.queue.clear()
            server.running.clear()
        for task in self.tasks.values():
            if task.status is not TaskStatus.FINISHED:
                task.status = TaskStatus.FAILED
                task.failed_at = self.deadline
        for dag in self.dag_runtimes.values():
            dag.mark_failed(self.deadline)
        self._result = SlotResult(
            slot_start=self.slot_start,
            slot_end=self.deadline,
            dags=tuple(DAGResult(key, dag.completion_time, dag.failed_at is not None)
                       for key, dag in sorted(self.dag_runtimes.items())),
            compute_energy_j=sum(task.compute_energy_j for task in self.tasks.values()),
            tx_energy_j=sum(record.tx_energy_j for task in self.tasks.values()
                            for record in (task.input_record, *task.predecessor_records.values())
                            if record is not None),
        )
        self._events.clear()
        self._ready_for_server.clear()
        self.channels.clear()
        self._transfer_channel.clear()
        self._transfer_bindings.clear()
        self._outgoing_transfers.clear()
        self._local_successors.clear()
        self._closed = True

    def trace(self, task_key: TaskKey) -> TaskRuntime:
        return self.tasks[task_key]

    def channel_state(self, key: DirectedChannelKey) -> DirectedChannelState:
        return self.channels[key]

    def update_entity_positions(self, positions: Mapping[EntityRef, np.ndarray]) -> None:
        """仅在当前时隙提交任务前更新实体位置快照。"""
        for entity, position in positions.items():
            position = np.asarray(position, dtype=float)
            self.entity_positions[entity] = position.copy()

    def update_topology(
        self,
        member_regions: Mapping[EntityRef, int],
        ground_owner_members: Mapping[EntityRef, EntityRef],
    ) -> None:
        """仅在当前时隙提交任务前刷新 Member 区域和用户关联。"""
        self.member_regions = {member: int(region_id) for member, region_id in member_regions.items()}
        self.ground_owner_members = dict(ground_owner_members)

    def _append_route(
        self,
        pending: list[tuple[DirectedChannelKey, TransferJob, _TransferBinding]],
        hops: tuple[DirectedChannelKey, ...],
        task: TaskRuntime,
        payload_bits: float,
        source_ready_at: float | None,
        transfer_counter: count,
        is_input: bool = False,
        predecessor: TaskKey | None = None,
    ) -> TransferRecord:
        transfer_ids = tuple(f"tx-{next(transfer_counter)}" for _ in hops)
        record = TransferRecord(transfer_ids)
        for index, hop in enumerate(hops):
            transfer_id = transfer_ids[index]
            job = TransferJob(
                transfer_id=transfer_id,
                priority_seq=task.priority_seq,
                source_ready_at=source_ready_at if index == 0 else None,
                duration_s=self.transfer_duration_s(hop, payload_bits),
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

    def transfer_duration_s(self, hop: DirectedChannelKey, payload_bits: float) -> float:
        """当前物理快照上的单跳时延，不含排队；排序和实际执行共用。"""
        transmitter = self.entity_positions[hop.source]
        receiver = self.entity_positions[hop.target]
        power = self.transmit_powers[hop.source]
        bandwidth_hz = self.directed_bandwidth_hz.get(hop, self._default_bandwidth_hz(hop))
        metrics = self.channel_model.calculate_link(
            transmitter, receiver, power, self._link_type(hop), bandwidth_hz
        )
        return float(payload_bits) / metrics.rate_bps

    def candidate_execution_nodes(
        self, owner_member: EntityRef, ground_device: EntityRef,
    ) -> tuple[EntityRef, ...]:
        """自身 Ground、当前同区域 Member 和唯一 BS，供排序与动作选择。"""
        region = self.member_regions[owner_member]
        members = sorted(member for member, region_id in self.member_regions.items() if region_id == region)
        return (ground_device, *members, self.global_bs)

    def _default_bandwidth_hz(self, hop: DirectedChannelKey) -> float:
        config = self.channel_model.config
        return {
            LinkType.GROUND_TO_AIR: config.ground_to_air_bandwidth_mhz,
            LinkType.AIR_TO_AIR: config.air_to_air_bandwidth_mhz,
            LinkType.AIR_TO_GROUND: config.air_to_ground_bandwidth_mhz,
        }[self._link_type(hop)] * 1e6

    @staticmethod
    def _link_type(hop: DirectedChannelKey) -> LinkType:
        if hop.source.kind is EntityKind.MEMBER_UAV and hop.target.kind is EntityKind.MEMBER_UAV:
            return LinkType.AIR_TO_AIR
        if hop.source.kind is EntityKind.MEMBER_UAV:
            return LinkType.AIR_TO_GROUND
        return LinkType.GROUND_TO_AIR

    def _handle_transfer_finish(self, transfer_id: str, at: float) -> None:
        state = self.channels[self._transfer_channel[transfer_id]]
        state.finish_active(at)
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
        if at >= self.deadline:
            return
        for task_key in sorted(
            self._ready_for_server,
            key=lambda key: (self.tasks[key].data_ready_at, self.tasks[key].priority_seq, key),
        ):
            task = self.tasks[task_key]
            self.servers[task.execution_node].enqueue(
                task.key, task.cpu_cycles, task.data_ready_at, task.priority_seq
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
