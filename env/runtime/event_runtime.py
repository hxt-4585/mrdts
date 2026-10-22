"""单个时隙内多 DAG 传输与计算的离散事件运行时。"""

from collections import defaultdict
from dataclasses import dataclass
import heapq
from itertools import count
from typing import Callable, Iterable, Mapping

import numpy as np

from env.communication.channel_model import ChannelModel, LinkType
from env.communication.routing import RoutePlanner
from env.contracts import DAGRequest, PlacementDecision
from env.settings import EnvConfig
from env.runtime.channel_queue import DirectedChannelState, TransferJob
from env.runtime.dag_runtime import DAGRuntime
from env.runtime.server_queue import ComputeRecord, ServerState
from env.runtime.slot_result import DAGResult, MemberEnergyResult, SlotResult
from env.runtime.task_runtime import TaskRuntime, TaskStatus, TransferRecord
from env.types import DirectedChannelKey, EntityKind, EntityRef, TaskKey


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
    """执行在线卸载动作；任务顺序和策略均由调用方持有。

    调用约定：先 register_dags，再为 ready_member_ids 中每个 Member
    选择一个未调度且前驱已完成的子任务，统一 submit_placements。
    每个 Member 最多一个活动子任务；推进到下一个决策点后重复。
    """

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
        max_duration_s: float | None = None,
        on_member_energy: Callable[[EntityRef, float], None] | None = None,
        on_time_advance: Callable[[float], None] | None = None,
    ):
        self._slot_start = float(slot_start)
        duration = float(max_duration_s if max_duration_s is not None else EnvConfig.default().scheduling_duration_s)
        self._deadline = self._slot_start + duration
        self._closed = False
        self._result: SlotResult | None = None
        # 环境接收事件增量以更新队列；独立运行时也保留逐 Member 结算明细。
        self._on_member_energy = on_member_energy
        self._on_time_advance = on_time_advance
        self._member_energy = {
            node: [0.0, 0.0] for node in servers if node.kind is EntityKind.MEMBER_UAV
        }
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
        self.transfer_jobs: dict[str, TransferJob] = {}
        self._requests: dict[tuple[int, int, int], DAGRequest] = {}
        self.active_tasks: dict[int, TaskKey] = {}
        self._remaining_tasks: dict[int, int] = defaultdict(int)
        self._ready_members: set[int] = set()
        self._ready_for_server: set[TaskKey] = set()
        self._events: list[tuple[float, int, str, object]] = []
        self._event_counter = count()
        self._transfer_counter = count()
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

    @property
    def ready_member_ids(self) -> tuple[int, ...]:
        """当前可接受下一次决策的 Member；不替 method 选择任务。"""
        return tuple(sorted(self._ready_members))

    def register_dags(self, requests: Iterable[DAGRequest]) -> tuple[TaskKey, ...]:
        """登记窗口工作负载，不确定位置、不发送数据、不预占资源。"""
        keys = []
        for request in sorted(requests, key=lambda item: item.key):
            self._requests[request.key] = request
            self.dag_runtimes[request.key] = DAGRuntime(request.dag)
            node_keys = request.task_keys
            predecessors = {key: set() for key in node_keys}
            for parent, child in request.dag.edges:
                predecessors[node_keys[child]].add(node_keys[parent])
            for key in node_keys:
                features = request.dag.node_features[key.node_id]
                self.tasks[key] = TaskRuntime(
                    key=key, cpu_cycles=float(features[1]), input_bits=float(features[0]) * 1000.,
                    predecessors=predecessors[key],
                )
            member_id = request.owner_member.index
            self._remaining_tasks[member_id] += len(node_keys)
            if node_keys and member_id not in self.active_tasks:
                self._ready_members.add(member_id)
            keys.extend(node_keys)
        return tuple(keys)

    def submit_placements(self, placements: Mapping[TaskKey, PlacementDecision]) -> None:
        """提交当前决策批次，每个可决策 Member 至多一个任务。

        调用方先基于同一时刻的状态完成本批全部观测与策略选择，再提交。
        TaskKey 仅用于同时提交时的稳定平局顺序，不代表策略优先级。
        """
        for key in sorted(placements):
            task = self.tasks[key]
            request = self._requests[key.owner_member_id, key.user_id, key.dag_id]
            task.execution_node = placements[key].execution_node
            task.decision_at = self.now
            task.status = TaskStatus.WAITING_DATA
            self.active_tasks[key.owner_member_id] = key
            self._ready_members.remove(key.owner_member_id)
            route = self.route_planner.input_route(
                request.ground_device, request.owner_member, task.execution_node)
            task.input_record = self._submit_data(task, route.hops, task.input_bits, is_input=True)
            for parent_key in sorted(task.predecessors):
                parent = self.tasks[parent_key]
                route = self.route_planner.predecessor_route(
                    parent.execution_node, task.execution_node, request.ground_device, request.owner_member)
                bits = float(request.dag.edge_features[parent_key.node_id, key.node_id]) * 1000.
                task.predecessor_records[parent_key] = self._submit_data(
                    task, route.hops, bits, predecessor=parent_key)
        self._dispatch(self.now)
        # 零时长传输/计算也须完成当前时刻的事件后才能重新观测。
        self._process_events_until(self.now)

    def _submit_data(
        self, task: TaskRuntime, hops: tuple[DirectedChannelKey, ...], bits: float,
        *, is_input: bool = False, predecessor: TaskKey | None = None,
    ) -> TransferRecord:
        if not hops or bits == 0:
            record = TransferRecord((), start_at=self.now, finish_at=self.now)
            ready = (task.mark_input_arrived(self.now) if is_input
                     else task.mark_predecessor_arrived(predecessor, self.now))
            if ready:
                self._ready_for_server.add(task.key)
            return record
        return self._append_route(hops, task, bits, is_input=is_input, predecessor=predecessor)

    def advance_to_decision(self) -> tuple[int, ...]:
        """推进至最早的下一次决策机会、全部完成或窗口截止。"""
        while self._events and not self._ready_members and not self.closed:
            self.advance_until(self._events[0][0])
        return self.ready_member_ids

    def advance_until(self, absolute_time: float) -> tuple[int, ...]:
        """推进至给定上限；出现可决策 Member 时立即停下，不越过决策点。"""
        limit = min(float(absolute_time), self.deadline)
        while not self.closed and not self._ready_members:
            if not self._events:
                self._advance_clock(limit)
                if limit == self.deadline:
                    self._close_slot()
                break
            timestamp = min(self._events[0][0], limit)
            self._process_events_until(timestamp)
            if self.now == self.deadline:
                self._close_slot()
            if timestamp == limit:
                break
        return self.ready_member_ids

    def _process_events_until(self, absolute_time: float) -> None:
        while self._events and self._events[0][0] <= absolute_time:
            timestamp = self._events[0][0]
            self._advance_clock(timestamp)
            # 先处理这个时刻已有的全部完成事件，再派发；即时事件继续闭包处理。
            while self._events and self._events[0][0] == timestamp:
                _, _, event_type, payload = heapq.heappop(self._events)
                if event_type == "compute_finish":
                    self._handle_compute_finish(payload, timestamp)
                else:
                    self._handle_transfer_finish(payload, timestamp)
            self._dispatch(timestamp)
        self._advance_clock(absolute_time)

    def _advance_clock(self, at: float) -> None:
        elapsed = at - self.now
        self.now = at
        if elapsed and self._on_time_advance is not None:
            self._on_time_advance(elapsed)

    def _record_member_energy(self, node: EntityRef, amount: float, component: int) -> None:
        """component 为 0（计算）或 1（发送），调用方仅传入 Member。"""
        self._member_energy[node][component] += amount
        if self._on_member_energy is not None:
            self._on_member_energy(node, amount)

    def _settle_transfer_energy(self, transfer_id: str, elapsed: float) -> None:
        source = self._transfer_channel[transfer_id].source
        if source.kind is EntityKind.MEMBER_UAV:
            binding = self._transfer_bindings[transfer_id]
            energy = binding.power_w * elapsed
            binding.record.tx_energy_j += energy
            self._record_member_energy(source, energy, 1)

    def _settle_compute_energy(self, node: EntityRef, record: ComputeRecord, energy: float) -> None:
        if node.kind is EntityKind.MEMBER_UAV:
            self.tasks[record.task_key].compute_energy_j = energy
            self._record_member_energy(node, energy, 0)

    def finish_slot(self) -> SlotResult:
        """截止结算；不再索取动作，尚未提交的任务也计入失败。"""
        if not self.closed:
            self._process_events_until(self.deadline)
            self._close_slot()
        return self._result

    def _close_slot(self) -> None:
        # 已完成作业已按事件结算；这里只补记仍在运行的作业在截止前的部分。
        for channel in self.channels.values():
            job = channel.active
            if job is not None:
                self._settle_transfer_energy(job.transfer_id, self.deadline - job.start_at)
            channel.queued.clear()
            channel.active = None
        for server in self.servers.values():
            for record in server.running.values():
                elapsed = self.deadline - record.start_at
                duration = record.finish_at - record.start_at
                self._settle_compute_energy(server.node, record, record.energy_j * elapsed / duration)
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
            compute_energy_j=sum(energy[0] for energy in self._member_energy.values()),
            tx_energy_j=sum(energy[1] for energy in self._member_energy.values()),
            member_energy=tuple(MemberEnergyResult(node.index, *energy)
                                for node, energy in sorted(self._member_energy.items())),
        )
        self._events.clear()
        self._ready_for_server.clear()
        self.channels.clear()
        self._transfer_channel.clear()
        self._transfer_bindings.clear()
        self.active_tasks.clear()
        self._ready_members.clear()
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
        self, hops: tuple[DirectedChannelKey, ...], task: TaskRuntime, payload_bits: float,
        *, is_input: bool = False, predecessor: TaskKey | None = None,
    ) -> TransferRecord:
        transfer_ids = tuple(f"tx-{next(self._transfer_counter)}" for _ in hops)
        record = TransferRecord(transfer_ids)
        for index, (hop, transfer_id) in enumerate(zip(hops, transfer_ids)):
            self.transfer_jobs[transfer_id] = TransferJob(
                transfer_id=transfer_id, source_ready_at=None,
                duration_s=self.transfer_duration_s(hop, payload_bits),
            )
            self._transfer_channel[transfer_id] = hop
            self._transfer_bindings[transfer_id] = _TransferBinding(
                task.key, record, self.transmit_powers[hop.source],
                next_transfer_id=transfer_ids[index + 1] if index + 1 < len(hops) else None,
                predecessor=predecessor, is_input=is_input,
            )
        self._activate_transfer(transfer_ids[0], self.now)
        return record

    def _activate_transfer(self, transfer_id: str, at: float) -> None:
        job = self.transfer_jobs[transfer_id]
        job.source_ready_at = at
        key = self._transfer_channel[transfer_id]
        state = self.channels.get(key)
        if state is None:
            state = self.channels[key] = DirectedChannelState(key)
        state.enqueue(job)

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
        job = state.finish_active(at)
        self._settle_transfer_energy(transfer_id, at - job.start_at)
        binding = self._transfer_bindings[transfer_id]
        if binding.next_transfer_id is not None:
            self._activate_transfer(binding.next_transfer_id, at)
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
        self._settle_compute_energy(server_node, record, record.energy_j)
        task = self.tasks[record.task_key]
        task.mark_compute_finished(at)
        dag_key = (task.key.owner_member_id, task.key.user_id, task.key.dag_id)
        self.dag_runtimes[dag_key].mark_finished(task.key.node_id, at)
        member_id = task.key.owner_member_id
        del self.active_tasks[member_id]
        self._remaining_tasks[member_id] -= 1
        if self._remaining_tasks[member_id] and at < self.deadline:
            self._ready_members.add(member_id)

    def _dispatch(self, at: float) -> None:
        if at >= self.deadline:
            return
        for task_key in sorted(
            self._ready_for_server,
            key=lambda key: (self.tasks[key].data_ready_at, key),
        ):
            task = self.tasks[task_key]
            self.servers[task.execution_node].enqueue(
                task.key, task.cpu_cycles, task.data_ready_at
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
            heapq.heappush(
                self._events, (job.finish_at, next(self._event_counter), "transfer_finish", job.transfer_id)
            )
        for server in self.servers.values():
            for record in server.dispatch(at):
                task = self.tasks[record.task_key]
                task.mark_compute_started(record.core_id, at)
                heapq.heappush(
                    self._events,
                    (record.finish_at, next(self._event_counter), "compute_finish", (server.node, record.core_id)),
                )
