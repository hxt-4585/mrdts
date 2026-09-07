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
from env.runtime.dag_runtime import DAGRuntime, validate_dag
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
        if (not np.isfinite(self._slot_start) or self._slot_start < 0.0
                or not np.isfinite(duration) or duration <= 0.0
                or not np.isfinite(self._deadline) or self._deadline <= self._slot_start):
            raise ValueError("时隙起点必须有限非负，窗口长度必须有限为正")
        self._closed = False
        self._result: SlotResult | None = None
        self.channel_model = channel_model
        self.entity_positions = {
            entity: np.array(position, dtype=float, copy=True) for entity, position in entity_positions.items()
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
        self._next_priority_seq = 0
        self.now = self.slot_start
        if member_regions is not None or ground_owner_members is not None:
            if member_regions is None or ground_owner_members is None:
                raise ValueError("member_regions 与 ground_owner_members 必须同时提供")
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

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("当前时隙运行时已关闭")

    def _require_mutable_topology(self) -> None:
        self._require_open()
        if self.tasks or self.now != self.slot_start:
            raise RuntimeError("提交任务或推进时间后不能修改时隙拓扑")

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
        self._require_open()
        if set(placements) != set(range(dag.node_num)):
            raise ValueError("每个子任务必须恰有一个 placement")
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
        self._require_open()
        epoch_start = float(epoch_start)
        if not np.isfinite(epoch_start) or not self.slot_start <= epoch_start < self.deadline:
            raise ValueError("DAG 提交时刻必须位于当前时隙窗口内")
        if epoch_start != self.now:
            raise ValueError("DAG 提交时刻必须等于当前运行时刻；请先推进到提交时刻")
        requests = tuple(requests)
        if len({request.key for request in requests}) != len(requests):
            raise ValueError("同一 owner/user/dag_id 的 DAG 重复")
        for request in requests:
            validate_dag(request.dag)
            if request.key in self.dag_runtimes:
                raise ValueError("同一 owner/user/dag_id 的 DAG 已提交")
        required = {key for request in requests for key in request.task_keys}
        if set(placements) != required:
            raise ValueError("每个子任务必须恰有一个 placement")
        sequences = [decision.priority_seq for decision in placements.values()]
        if any(isinstance(seq, bool) or not isinstance(seq, int) or seq < 0 for seq in sequences):
            raise ValueError("priority_seq 必须为非负整数")
        if len(set(sequences)) != len(sequences):
            raise ValueError("同一批次的 priority_seq 不能重复")
        for request in requests:
            decisions = {key.node_id: placements[key] for key in request.task_keys}
            self._validate_submission_topology(request.owner_member, request.ground_device, decisions)
            for parent, child in request.dag.edges:
                if decisions[parent].priority_seq >= decisions[child].priority_seq:
                    raise ValueError("优先顺序必须使前驱先于后继，避免信道队首阻塞死锁")
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
            if decision.execution_node not in self.servers:
                raise ValueError(f"执行节点 {decision.execution_node} 没有计算服务器")
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
        self._require_open()
        absolute_time = float(absolute_time)
        if not np.isfinite(absolute_time) or absolute_time > self.deadline:
            raise ValueError("推进时刻必须有限且不能超过当前时隙截止时刻")
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
        if self.now == self.deadline:
            self._close_slot()

    def finish_slot(self) -> SlotResult:
        """执行到截止时刻并关闭；重复调用只返回同一份结果。"""
        if not self.closed:
            self.advance_until(self.deadline)
        assert self._result is not None
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
        self._require_mutable_topology()
        for entity, position in positions.items():
            if entity not in self.entity_positions:
                raise KeyError(f"运行时不认识实体 {entity}")
            position = np.asarray(position, dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("实体位置必须是包含有限数值的 shape=(3,) 向量")
            self.entity_positions[entity] = position.copy()

    def update_topology(
        self,
        member_regions: Mapping[EntityRef, int],
        ground_owner_members: Mapping[EntityRef, EntityRef],
    ) -> None:
        """仅在当前时隙提交任务前刷新 Member 区域和用户关联。"""
        self._require_mutable_topology()
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
        self._require_open()
        if not np.isfinite(payload_bits) or payload_bits < 0:
            raise ValueError("传输数据量必须为有限非负数")
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
        self._require_open()
        if self.member_regions is None or self.ground_owner_members is None:
            raise RuntimeError("提交 DAG 前必须提供当前时隙拓扑")
        if owner_member.kind is not EntityKind.MEMBER_UAV or owner_member not in self.member_regions:
            raise ValueError("owner_member 必须是已登记的 Member UAV")
        if ground_device.kind is not EntityKind.GROUND_DEVICE or ground_device not in self.ground_owner_members:
            raise ValueError("ground_device 必须是已登记的地面设备")
        if self.ground_owner_members[ground_device] != owner_member:
            raise ValueError("地面设备当前关联的 Member UAV 与 owner_member 不一致")
        region = self.member_regions[owner_member]
        members = sorted(member for member, region_id in self.member_regions.items() if region_id == region)
        return (ground_device, *members, self.global_bs)

    def _validate_submission_topology(
        self,
        owner_member: EntityRef,
        ground_device: EntityRef,
        placements: Mapping[int, PlacementDecision],
    ) -> None:
        candidates = self.candidate_execution_nodes(owner_member, ground_device)
        for decision in placements.values():
            execution_node = decision.execution_node
            if execution_node.kind is EntityKind.GROUND_DEVICE:
                if execution_node != ground_device:
                    raise ValueError("子任务只能在自身地面设备本地执行")
            elif execution_node.kind is EntityKind.MEMBER_UAV:
                if execution_node not in candidates:
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
