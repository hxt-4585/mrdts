"""单轮多 DAG 传输与计算的离散事件运行时；执行不受名义时隙长度限制。"""

from collections import Counter, deque
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
from env.runtime.communication_resources import TransferAllocation, transfer_allocations
from env.runtime.dag_runtime import DAGRuntime
from env.runtime.server_queue import ComputeRecord, ServerState
from env.runtime.slot_result import DAGResult, MemberEnergyResult, SlotResult
from env.runtime.task_runtime import TaskRuntime, TaskStatus, TransferRecord
from env.step.reward import MemberReward
from env.types import DirectedChannelKey, EntityKind, EntityRef, TaskKey


@dataclass(frozen=True)
class ServerSpec:
    core_frequencies: tuple[float, ...]
    capacitance_factor: float


@dataclass
class _TransferBinding:
    task_key: TaskKey
    record: TransferRecord
    next_transfer_id: str | None = None
    predecessor: TaskKey | None = None
    is_input: bool = False


class SchedulingRuntime:
    """执行整窗放置方案；原始输入在窗口开始就绪，前驱结果在计算完成时就绪。"""

    def __init__(
        self,
        channel_model: ChannelModel,
        entity_positions: Mapping[EntityRef, np.ndarray],
        servers: Mapping[EntityRef, ServerSpec],
        transmit_powers: Mapping[EntityRef, float],
        member_regions: Mapping[EntityRef, int] | None = None,
        *,
        slot_start: float = 0.0,
        max_duration_s: float | None = None,
        on_member_energy: Callable[[EntityRef, float], None] | None = None,
        on_time_advance: Callable[[float], None] | None = None,
    ):
        self._slot_start = float(slot_start)
        duration = float(max_duration_s if max_duration_s is not None else EnvConfig.default().scheduling_duration_s)
        self._deadline = self._slot_start + duration
        self._scheduling_duration_s = duration
        self._placement_order: tuple[TaskKey, ...] = ()
        self._member_rewards: dict[TaskKey, MemberReward] = {}
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
        self._channel_gains: dict[DirectedChannelKey, float] = {}
        self._active_transfers: dict[DirectedChannelKey, TransferJob] = {}
        self._communication_changed = False
        self.member_regions: dict[EntityRef, int] | None = None
        self.route_planner = RoutePlanner()
        self.channels: dict[DirectedChannelKey, DirectedChannelState] = {}
        self.tasks: dict[TaskKey, TaskRuntime] = {}
        self.dag_runtimes: dict[tuple[int, int, int], DAGRuntime] = {}
        self._transfer_channel: dict[str, DirectedChannelKey] = {}
        self._transfer_bindings: dict[str, _TransferBinding] = {}
        self.transfer_jobs: dict[str, TransferJob] = {}
        self._requests: dict[tuple[int, int, int], DAGRequest] = {}
        self._ready_for_server: set[TaskKey] = set()
        self._events: list[tuple[float, int, str, object]] = []
        self._event_counter = count()
        self._transfer_counter = count()
        self.now = self.slot_start
        if member_regions is not None:
            self.update_topology(member_regions)

    @property
    def slot_start(self) -> float:
        return self._slot_start

    @property
    def deadline(self) -> float:
        """名义时隙边界，仅作时间参考，不截断执行。"""
        return self._deadline

    @property
    def closed(self) -> bool:
        return self._closed

    def register_dags(self, requests: Iterable[DAGRequest]) -> tuple[TaskKey, ...]:
        """登记窗口工作负载；所有传输与计算等待整窗方案提交。"""
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
                    member_energy_j=np.zeros(len(self._member_energy)),
                )
            keys.extend(node_keys)
        return tuple(keys)

    def submit_placements(self, placements: dict[TaskKey, PlacementDecision]) -> None:
        """按 method 给出的调度顺序登记整窗方案。无关联任务自动放在源设备。"""
        self._placement_order = tuple(placements)
        for index, (key, decision) in enumerate(placements.items()):
            task = self.tasks[key]
            task.schedule_index = index
            task.execution_node = decision.execution_node
        owner_counts = Counter(request.owner_member for request in self._requests.values()
                               if request.owner_member is not None)

        for key, task in self.tasks.items():
            request = self._requests[key.owner_member_id, key.user_id, key.dag_id]
            owner = request.owner_member
            hops = ()
            if owner is None:
                task.execution_node = request.ground_device
            else:
                task.decision_at = self.now
                task.unfinished_dags_at_decision = owner_counts[owner]
                hops = self.route_planner.input_route(request.ground_device, owner, task.execution_node).hops
            task.status = TaskStatus.WAITING_DATA
            task.input_record = self._submit_data(task, hops, task.input_bits, is_input=True)
        self._dispatch(self.now)
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

    def advance_until(self, absolute_time: float) -> None:
        """执行至指定仿真时刻，包括该时刻的全部完成事件。"""
        self._process_events_until(absolute_time)

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

    def _record_member_rewards(self) -> None:
        """按提交顺序，将每个负责 Member 的最晚完工时间增量分给对应节点。"""
        latest_by_owner: dict[int, float] = {}
        for key in self._placement_order:
            task = self.tasks[key]
            owner = key.owner_member_id
            previous = latest_by_owner.get(owner, self.slot_start)
            latest = max(previous, task.compute_finish_at)
            latest_by_owner[owner] = latest
            self._member_rewards[key] = MemberReward(
                (previous - latest) / self._scheduling_duration_s, task.member_energy_j,
            )

    def _advance_clock(self, at: float) -> None:
        elapsed = at - self.now
        self.now = at
        if elapsed:
            for job in self._active_transfers.values():
                # 浮点舍入可能使恰好完成时的剩余量略小于零。
                job.remaining_bits = max(0.0, job.remaining_bits - job.rate_bps * elapsed)
                self._settle_transfer_energy(job.transfer_id, elapsed)
        if elapsed and self._on_time_advance is not None:
            self._on_time_advance(elapsed)

    def _record_member_energy(self, node: EntityRef, amount: float, component: int) -> None:
        """component 为 0（计算）或 1（发送），调用方仅传入 Member。"""
        self._member_energy[node][component] += amount
        if self._on_member_energy is not None:
            self._on_member_energy(node, amount)

    def _settle_transfer_energy(self, transfer_id: str, elapsed: float) -> None:
        job = self.transfer_jobs[transfer_id]
        energy = job.power_w * elapsed
        job.tx_energy_j += energy
        source = self._transfer_channel[transfer_id].source
        if source.kind is EntityKind.MEMBER_UAV:
            binding = self._transfer_bindings[transfer_id]
            binding.record.tx_energy_j += energy
            self.tasks[binding.task_key].member_energy_j[source.index] += energy
            self._record_member_energy(source, energy, 1)

    def _settle_compute_energy(self, node: EntityRef, record: ComputeRecord, energy: float) -> None:
        if node.kind is EntityKind.MEMBER_UAV:
            self.tasks[record.task_key].compute_energy_j = energy
            self.tasks[record.task_key].member_energy_j[node.index] += energy
            self._record_member_energy(node, energy, 0)

    def finish_slot(self) -> SlotResult:
        """执行整窗放置方案至全部事件完成并结算。

        名义边界不限制执行。提前完成时结算至名义边界，超出时使用实际结束时刻。
        """
        if not self.closed:
            while self._events:
                self._process_events_until(self._events[0][0])
            self._advance_clock(max(self.now, self.deadline))
            self._close_slot()
        return self._result

    def _close_slot(self) -> None:
        dags = []
        for key, dag in sorted(self.dag_runtimes.items()):
            completion_time = dag.completion_time
            # 超时只影响结算结果，不中断执行；恰好到达边界仍算成功。
            failed = (
                completion_time is not None and bool(completion_time > self.deadline)
            )
            dags.append(DAGResult(key, completion_time, failed))
        self._record_member_rewards()
        self._result = SlotResult(
            slot_start=self.slot_start,
            slot_end=self.now,
            dags=tuple(dags),
            compute_energy_j=sum(energy[0] for energy in self._member_energy.values()),
            tx_energy_j=sum(energy[1] for energy in self._member_energy.values()),
            member_energy=tuple(MemberEnergyResult(node.index, *energy)
                                for node, energy in sorted(self._member_energy.items())),
        )
        self._events.clear()
        self._ready_for_server.clear()
        self.channels.clear()
        self._active_transfers.clear()
        self._transfer_channel.clear()
        self._transfer_bindings.clear()
        self._closed = True

    def trace(self, task_key: TaskKey) -> TaskRuntime:
        return self.tasks[task_key]

    def pop_member_rewards(self) -> dict[TaskKey, MemberReward]:
        """领取窗口结算后的逐节点奖励；每个 TaskKey 仅返回一次。"""
        rewards = self._member_rewards
        self._member_rewards = {}
        return rewards

    def update_topology(
        self,
        member_regions: Mapping[EntityRef, int],
    ) -> None:
        """仅在当前时隙提交任务前刷新 Member 区域。"""
        self.member_regions = {member: int(region_id) for member, region_id in member_regions.items()}

    def _append_route(
        self, hops: tuple[DirectedChannelKey, ...], task: TaskRuntime, payload_bits: float,
        *, is_input: bool = False, predecessor: TaskKey | None = None,
    ) -> TransferRecord:
        transfer_ids = tuple(f"tx-{next(self._transfer_counter)}" for _ in hops)
        record = TransferRecord(transfer_ids)
        for index, (hop, transfer_id) in enumerate(zip(hops, transfer_ids)):
            self.transfer_jobs[transfer_id] = TransferJob(
                transfer_id=transfer_id, source_ready_at=None,
                payload_bits=payload_bits, channel_gain=self._channel_gain(hop),
            )
            self._transfer_channel[transfer_id] = hop
            self._transfer_bindings[transfer_id] = _TransferBinding(
                task.key, record,
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
        task = self.tasks[self._transfer_bindings[transfer_id].task_key]
        state.enqueue(job, task.schedule_index)
        if state.ready_head(at) is job:
            self._communication_changed = True

    def transfer_duration_s(self, hop: DirectedChannelKey, payload_bits: float) -> float:
        """ERS 的独占资源参考时延；实际执行由共享资源和剩余数据决定。"""
        rate = self.channel_model.rate_bps(
            self._channel_gain(hop), self.transmit_powers[hop.source],
            self.channel_model.config.total_bandwidth_mhz * 1e6,
        )
        return float(payload_bits) / rate

    def _channel_gain(self, hop: DirectedChannelKey) -> float:
        if hop not in self._channel_gains:
            self._channel_gains[hop] = self.channel_model.calculate_link(
                self.entity_positions[hop.source], self.entity_positions[hop.target],
                self.transmit_powers[hop.source], self._link_type(hop),
            ).average_channel_gain
        return self._channel_gains[hop]

    def candidate_execution_nodes(
        self, owner_member: EntityRef, ground_device: EntityRef,
    ) -> tuple[EntityRef, ...]:
        """自身 Ground、当前同区域 Member 和唯一 BS，供排序与动作选择。"""
        region = self.member_regions[owner_member]
        members = sorted(member for member, region_id in self.member_regions.items() if region_id == region)
        return (ground_device, *members, self.global_bs)

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
        del self._active_transfers[state.key]
        self._communication_changed = True
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
        record = self.servers[server_node].finish(core_id)
        self._settle_compute_energy(server_node, record, record.energy_j)
        task = self.tasks[record.task_key]
        task.mark_compute_finished(at)
        dag_key = (task.key.owner_member_id, task.key.user_id, task.key.dag_id)
        dag = self.dag_runtimes[dag_key]
        dag.mark_finished(task.key.node_id, at)
        request = self._requests[dag_key]
        for child_id in sorted(dag.successors[task.key.node_id]):
            child = self.tasks[TaskKey(*dag_key, child_id)]
            route = self.route_planner.predecessor_route(
                task.execution_node, child.execution_node, request.ground_device, request.owner_member)
            bits = float(request.dag.edge_features[task.key.node_id, child_id]) * 1000.
            child.predecessor_records[task.key] = self._submit_data(
                child, route.hops, bits, predecessor=task.key)

    def _enqueue_ready_computes(self) -> None:
        for task_key in sorted(self._ready_for_server):
            task = self.tasks[task_key]
            self.servers[task.execution_node].enqueue(
                task.key, task.cpu_cycles, task.data_ready_at, task.schedule_index
            )
        self._ready_for_server.clear()

    def _dispatch(self, at: float) -> None:
        idle: dict[EntityRef, deque[int]] = {}
        allocations = None
        while True:
            if self._communication_changed:
                allocations = self._settle_instant_transfers(at)
            self._enqueue_ready_computes()
            candidates = []
            for node, server in self.servers.items():
                if server.queue:
                    if node not in idle:
                        idle[node] = deque(server.idle_cores())
                    if idle[node]:
                        head = server.queue[0]
                        candidates.append((head.ready_at, head.schedule_index, head.task_key, node))
            heapq.heapify(candidates)
            while candidates:
                _, _, key, node = heapq.heappop(candidates)
                server = self.servers[node]
                core = idle[node].popleft()
                record = server.start_next(core, at)
                task = self.tasks[key]
                task.mark_compute_started(core, at)
                if task.cpu_cycles == 0:
                    self._handle_compute_finish((node, core), at)
                    idle[node].append(core)
                    if self._ready_for_server or self._communication_changed:
                        # 即时释放的计算和传输须在下一次选择核心之前参与竞争。
                        break
                else:
                    heapq.heappush(self._events, (record.finish_at, next(self._event_counter),
                                                  "compute_finish", (node, core)))
                if server.queue and idle[node]:
                    head = server.queue[0]
                    heapq.heappush(candidates, (head.ready_at, head.schedule_index, head.task_key, node))
            else:
                if allocations is not None:
                    self._commit_transfers(allocations, at)
                return

    def _start_transfer(self, job: TransferJob, at: float) -> None:
        key = self._transfer_channel[job.transfer_id]
        if self.channels[key].active is None:
            self.channels[key].start_head(at)
            binding = self._transfer_bindings[job.transfer_id]
            if binding.record.start_at is None:
                binding.record.start_at = at
            self._active_transfers[key] = job

    def _settle_instant_transfers(self, at: float) -> list[TransferAllocation]:
        while True:
            candidates = self._active_transfers.copy()
            for key, state in self.channels.items():
                head = state.ready_head(at)
                if head is not None:
                    candidates[key] = head
            allocations = list(transfer_allocations(candidates, self.channel_model, self.transmit_powers, at))
            immediate = [item for item in allocations if item.finish_at == at]
            if not immediate:
                self._communication_changed = False
                return allocations
            # 先锁定这一批同时完成的队首，再释放下一跳，避免新就绪的作业改变本批队首。
            for item in immediate:
                self._start_transfer(item.job, at)
                item.apply()
            for item in immediate:
                self._handle_transfer_finish(item.job.transfer_id, at)

    def _commit_transfers(self, allocations: list[TransferAllocation], at: float) -> None:
        # 同刻完成事件已处理完毕，只把最终活动传输的未来完成事件加入事件堆。
        self._events = [event for event in self._events if event[2] != "transfer_finish"]
        for item in allocations:
            self._start_transfer(item.job, at)
            item.apply()
            self._events.append((item.finish_at, next(self._event_counter), "transfer_finish", item.job.transfer_id))
        heapq.heapify(self._events)
