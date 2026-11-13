"""MRDTS 环境入口；组织 epoch 内的飞行、调度、观测与奖励。"""

from collections.abc import Mapping
import random

import numpy as np

from env.communication.channel_model import ChannelModel
from env.contracts import BeginStepResult, DAGRequest, EndStepResult, PlacementDecision
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User
from env.runtime.event_runtime import SchedulingRuntime, ServerSpec
from env.step.observation import MasterObservation, MemberObservation
from env.types import EntityKind, EntityRef, TaskKey
from env.workload.dag_generator import DAGGenerator


class MRDTSEnv:
    """持有当前场景和时钟，向 method 提供分阶段的 step 接口。"""

    def __init__(self, settings: Mapping[str, object]):
        """保存环境配置；场景由 generate_scene 单独生成。"""
        self.settings = settings
        self.env_config = settings["env"]
        self.region = None
        self.users = None
        self.masters = None
        self.members = None
        self._channel_model = None
        self._master_observation = None
        self._member_observation = None
        self._initial_member_positions = None
        self._initial_member_region_ids = None
        self._initial_region_member_counts = None
        self.epoch_index = None
        self.step_index = 0
        self._next_slot_start = 0.0
        self._runtime = None
        self.user_member_ids = None
        self.flight_violations = None
        self.dag_generator = None
        self._flight_energy_j = None
        self._hover_energy_j = 0.0
        self._hover_power_w = None

    def generate_scene(self):
        """按 env_seed 生成区域、地面用户以及 Master/Member UAV。"""
        spatial_seeds = np.random.SeedSequence(
            [self.env_config.env_seed, 0]
        ).generate_state(2)

        region = Region(self.settings["region"], rng=np.random.default_rng(int(spatial_seeds[0])))
        region.generate()
        users = User(self.settings["user"], rng=np.random.default_rng(int(spatial_seeds[1])))
        users.generate_from_region(region)
        masters = MasterUAV(self.settings["uav"])
        masters.generate_from_region(region)
        members = MemberUAV(self.settings["uav"], self.env_config)
        members.generate_from_region_and_user(region, masters)

        self.region = region
        self.users = users
        self.masters = masters
        self.members = members
        self._hover_power_w = float(members.propulsion_power(0.0))
        self._channel_model = ChannelModel(self.settings["channel"])
        self._master_observation = MasterObservation(region, masters, members)
        self._member_observation = MemberObservation(region, members, self.settings["dag"])
        self._initial_member_positions = members.positions.copy()
        self._initial_member_region_ids = members.region_ids.copy()
        self._initial_region_member_counts = members.region_member_counts.copy()

    def reset(self, *, epoch_index: int):
        """回退 UAV 状态，清空时隙状态并重建本 epoch 的任务随机流。"""
        np.copyto(self.members.positions, self._initial_member_positions)
        np.copyto(self.members.region_ids, self._initial_member_region_ids)
        np.copyto(self.members.region_member_counts, self._initial_region_member_counts)
        self.members.energy_virtual_queue_j.fill(0.0)

        self.epoch_index = epoch_index
        self.step_index = 0
        self._next_slot_start = 0.0
        self._runtime = None
        self.user_member_ids = None
        self.flight_violations = None
        self._flight_energy_j = None
        self._hover_energy_j = 0.0

        workload_seed = int(
            np.random.SeedSequence([self.env_config.env_seed, epoch_index]).generate_state(3)[2]
        )
        self.dag_generator = DAGGenerator(
            self.settings["dag"],
            rng=np.random.default_rng(workload_seed),
            py_rng=random.Random(workload_seed),
        )

    def observe_master(self, master_id: int) -> np.ndarray:
        """返回指定 Master 的 (3M,) 观测，供高层决策使用。"""
        return self._master_observation.build(master_id)

    def observe_member(
        self, task_key: TaskKey, placements: Mapping[TaskKey, PlacementDecision],
    ) -> np.ndarray:
        """结合窗口快照和 method 已选的前驱位置生成规划观测。"""
        return self._member_observation.build(task_key, self._runtime, placements)

    def begin_step(self, flight_actions: np.ndarray) -> BeginStepResult:
        """应用 (M, 2) 飞行动作，登记本窗口用于规划的全部 DAG。

        每行动作为 [a_x, a_y]，两个分量各自位于 [-1, 1]。
        调用方先 reset，后续窗口须先结算上一窗口。每个用户生成一个 DAG；
        无 Member 区域的 DAG 自动本地执行，其余任务由 method 规划位置。
        飞行结束即记入推进能耗和能量虚拟队列；不推进调度时钟或 step_index。
        """
        previous_positions = self.members.positions[:self.members.member_uav_count, :2].astype(float)
        self.flight_violations = self.members.apply_flight_actions(
            flight_actions, self.region.config.side_length
        )
        velocity = (self.members.positions[:self.members.member_uav_count, :2] - previous_positions) / self.env_config.flight_duration_s
        speed = np.linalg.norm(velocity, axis=1)
        self._flight_energy_j = self.members.propulsion_power(speed) * self.env_config.flight_duration_s
        self.members.energy_virtual_queue_j += self._flight_energy_j
        self._hover_energy_j = 0.0
        self._refresh_topology()
        self._create_scheduling_runtime()
        requests = self._generate_dag_requests()
        self._runtime.register_dags(requests)
        return BeginStepResult(requests, self.flight_violations)

    def submit_schedule(
        self,
        placements: dict[TaskKey, PlacementDecision],
    ) -> None:
        """提交整窗位置。计算和每跳传输均按就绪时间排队，同一时刻就绪时按字典插入顺序派发。"""
        self._runtime.submit_placements(placements)

    def end_step(self) -> EndStepResult:
        """每步调用一次：完成执行并结算后扣除一次预算，返回快照并推进步数。

        已完成作业和飞行能耗已入账，finish_slot 执行剩余已提交作业并补计悬停。
        调度时钟不含飞行阶段；返回全部关联子任务的完成奖励。
        """
        slot_result = self._runtime.finish_slot()
        member_rewards = self._runtime.pop_member_rewards()
        count = self.members.member_uav_count
        compute_energy = np.zeros(count)
        tx_energy = np.zeros(count)
        for energy in slot_result.member_energy:
            compute_energy[energy.member_id] = energy.compute_energy_j
            tx_energy[energy.member_id] = energy.tx_energy_j
        flight_energy = self._flight_energy_j.copy()
        hover_energy = np.full(count, self._hover_energy_j)
        total_energy = compute_energy + tx_energy + flight_energy + hover_energy
        queues = self.members.energy_virtual_queue_j
        np.subtract(queues, self.members.config.member_slot_energy, out=queues)
        np.maximum(queues, 0., out=queues)
        self.step_index += 1
        self._next_slot_start = slot_result.slot_end
        self._runtime = None
        return EndStepResult(
            slot_result, compute_energy, tx_energy, flight_energy, hover_energy,
            total_energy, queues.copy(), self.step_index >= self.env_config.steps_per_epoch,
            scheduling_duration_s=self.env_config.scheduling_duration_s,
            member_rewards=member_rewards,
        )

    def _record_member_energy(self, node: EntityRef, energy_j: float) -> None:
        self.members.energy_virtual_queue_j[node.index] += energy_j

    def _record_hover_energy(self, elapsed_s: float) -> None:
        energy_j = self._hover_power_w * elapsed_s
        self._hover_energy_j += energy_j
        self.members.energy_virtual_queue_j += energy_j

    def _refresh_topology(self):
        """用户关联本区域最近的 Member；无 Member 时记为 -1，自动本地执行。"""
        member_count = self.members.member_uav_count
        member_regions = self.members.region_ids[:member_count]
        for member_id, position in enumerate(self.members.positions[:member_count]):
            member_regions[member_id] = self.region.get_region_id(*position[:2])
        self.members.region_member_counts[:] = np.bincount(
            member_regions, minlength=self.region.config.region_count + 1
        )[1:]

        region_members = tuple(
            np.flatnonzero(member_regions == region_id)
            for region_id in range(1, self.region.config.region_count + 1)
        )
        self.user_member_ids = np.full(self.users.num_users, -1, dtype=np.int32)
        for user_id, (position, region_id) in enumerate(zip(self.users.positions, self.users.region_ids)):
            candidates = region_members[region_id - 1]
            if not candidates.size:
                continue
            displacement = self.members.positions[candidates, :2] - position[:2]
            squared_distances = np.sum(displacement * displacement, axis=1)
            self.user_member_ids[user_id] = candidates[np.argmin(squared_distances)]

    def _create_scheduling_runtime(self):
        """按飞行后的拓扑创建本步调度运行时。"""
        self._member_observation.capture_initial_energy_queues()
        members = self.members
        member_nodes = tuple(
            EntityRef(EntityKind.MEMBER_UAV, member_id)
            for member_id in range(members.member_uav_count)
        )
        compute_nodes = (*member_nodes, EntityRef(EntityKind.BS, 0))
        positions = {}
        powers = {}
        servers = {}
        for node_id, node in enumerate(compute_nodes):
            positions[node] = members.positions[node_id]
            powers[node] = (members.config.bs_transmit_power if node.kind is EntityKind.BS
                            else members.config.member_transmit_power)
            servers[node] = ServerSpec(
                tuple(members.core_frequencies[node_id, :members.core_counts[node_id]]),
                members.config.member_capacitance_factor,
            )

        ground_server = ServerSpec((self.users.config.core_frequency,) * self.users.config.core_count, 0.0)
        for user_id, position in enumerate(self.users.positions):
            ground = EntityRef(EntityKind.GROUND_DEVICE, user_id)
            positions[ground] = position
            powers[ground] = self.users.config.transmit_power
            servers[ground] = ground_server

        self._runtime = SchedulingRuntime(
            self._channel_model, positions, servers, powers,
            member_regions={node: int(members.region_ids[node.index]) for node in member_nodes},
            slot_start=self._next_slot_start,
            max_duration_s=self.env_config.scheduling_duration_s,
            on_member_energy=self._record_member_energy,
            on_time_advance=self._record_hover_energy,
        )

    def _generate_dag_requests(self) -> tuple[DAGRequest, ...]:
        """为当前步生成 DAG 请求并绑定用户关联。"""
        member_nodes = tuple(
            EntityRef(EntityKind.MEMBER_UAV, member_id)
            for member_id in range(self.members.member_uav_count)
        )
        first_dag_id = self.step_index * self.users.num_users
        return tuple(
            DAGRequest(
                first_dag_id + user_id,
                self.dag_generator.generate_single_dag(),
                member_nodes[member_id] if member_id != -1 else None,
                EntityRef(EntityKind.GROUND_DEVICE, user_id),
            )
            for user_id, member_id in enumerate(self.user_member_ids)
        )
