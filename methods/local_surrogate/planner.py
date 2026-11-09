"""一跳观测快照、候选搜索和邻居标量报价。

只在 capture_views 读取环境；搜索器不持有环境、任务随机流或调度器。
邻居各自保有局部状态，对候选返回标量报价，不转发其邻居的原始状态。
单进程顺序模拟分布式协议，不声称已经实现并行加速或全局最优。
"""

from dataclasses import dataclass

import numpy as np

from methods.local_surrogate.surrogate import OnlineSurrogate


@dataclass
class LocalView:
    owned: np.ndarray
    visible: np.ndarray
    positions: np.ndarray
    queues: np.ndarray  # Q / E_budget，只有本区域及邻居可见
    visible_regions: tuple
    users: tuple  # 每个可见区域的固定 GD 位置
    previous_delays: np.ndarray  # 上一时隙区域总时延 / H
    region_map: np.ndarray  # 公共静态地图
    side_length: float
    full_step: float
    region_count: int

    def features(self, actions):
        single = actions.ndim == 2
        actions = np.atleast_3d(actions) if not single else actions[None]
        visible_actions = actions * self.visible[None, :, None]
        positions = self.positions[None] + self.full_step * visible_actions
        normalized = positions / self.side_length
        cells = np.minimum((normalized * len(self.region_map)).astype(int), len(self.region_map) - 1)
        regions = self.region_map[cells[:, :, 1], cells[:, :, 0]]
        k, members = actions.shape[:2]
        geometry = []
        for index, (region, users) in enumerate(zip(self.visible_regions, self.users)):
            present = (regions == region) & self.visible
            distances = ((normalized[:, :, None] - users[None, None] / self.side_length) ** 2).sum(-1)
            nearest = np.where(present[:, :, None], distances, 2.).min(axis=1).mean(axis=1)
            geometry.extend((present.sum(axis=1) / members, nearest,
                             np.full(k, len(users) / members),
                             np.full(k, self.previous_delays[index] / len(users))))
        features = np.column_stack((
            (normalized * self.visible[None, :, None]).reshape(k, -1),
            visible_actions.reshape(k, -1),
            np.broadcast_to(self.visible, (k, members)),
            np.broadcast_to(self.owned, (k, members)),
            regions * self.visible / self.region_count, *geometry))
        return features[0] if single else features


def capture_views(env, previous_delays):
    """把本区域遥测和一跳摘要投影到独立快照；不读取本时隙 DAG。"""
    count = env.members.member_uav_count
    views = []
    for master, region in enumerate(env.masters.region_ids):
        visible_regions = (int(region), *env.masters.neighbor_region_ids[int(region)])
        visible = np.isin(env.members.region_ids[:count], visible_regions)
        observation = env.observe_master(master).reshape(count, 3)
        users = tuple(env.users.positions[env.users.region_ids == r, :2].copy() for r in visible_regions)
        views.append(LocalView(
            owned=env.members.region_ids[:count] == region, visible=visible,
            positions=observation[:, :2].copy() * env.region.config.side_length,
            queues=observation[:, 2].copy(), visible_regions=visible_regions, users=users,
            previous_delays=previous_delays[np.array(visible_regions) - 1].copy(),
            region_map=env.region.region_map, side_length=env.region.config.side_length,
            full_step=env.members.config.max_horizontal_speed * env.env_config.flight_duration_s,
            region_count=env.region.config.region_count))
    return views


class LocalPlanner:
    def __init__(self, env, *, candidates=32, rounds=2, features=32, ridge=1.,
                 seed=45, delay_weight=1., uncertainty_penalty=.05):
        self.candidate_count = candidates
        self.rounds = rounds
        self.seed = seed
        self.delay_weight = delay_weight
        self.uncertainty_penalty = uncertainty_penalty
        self.count = env.members.member_uav_count
        self.full_step = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s
        self.speed_scale = env.members.config.max_horizontal_speed
        self.flight_duration = env.env_config.flight_duration_s
        self.energy_budget = env.members.config.member_slot_energy
        self.propulsion = env.members.propulsion_power
        self.regions = tuple(map(int, env.masters.region_ids))
        region_to_master = {region: i for i, region in enumerate(self.regions)}
        self.neighbors = [tuple(region_to_master[r] for r in env.masters.neighbor_region_ids[region])
                          for region in self.regions]
        views = capture_views(env, np.zeros(len(self.regions)))
        zero = np.zeros((self.count, 2))
        self.models = [OnlineSurrogate(len(view.features(zero)), 1 + self.count,
                       features=features, ridge=ridge, seed=seed + i)
                       for i, view in enumerate(views)]
        self.reset(0)

    def reset(self, episode):
        self.rngs = [np.random.default_rng(np.random.SeedSequence([self.seed, episode, i]))
                     for i in range(len(self.models))]

    def bounded(self, view, proposals):
        proposals[:, ~view.owned] = 0.
        lower = np.maximum(-1., -view.positions[view.owned] / self.full_step)
        upper = np.minimum(1., (view.side_length - view.positions[view.owned]) / self.full_step)
        proposals[:, view.owned] = np.clip(proposals[:, view.owned], lower, upper)
        return proposals

    def candidates(self, view, base, owner):
        rng = self.rngs[owner]
        proposals = np.broadcast_to(base, (self.candidate_count, *base.shape)).copy()
        local = rng.uniform(-1., 1., proposals.shape)
        local[0] = base
        local[1] = 0.
        # 第三个候选让各 Member 接近本区域离自身最近的 GD。
        users = view.users[0]
        nearest = ((view.positions[:, None] - users[None]) ** 2).sum(-1).argmin(axis=1)
        local[2] = np.clip((users[nearest] - view.positions) / self.full_step, -1., 1.)
        index = 3
        for users in view.users[1:]:
            center = users.mean(axis=0)
            for member in np.flatnonzero(view.owned):
                if index == self.candidate_count:
                    break
                local[index] = base
                local[index, member] = np.clip((center - view.positions[member]) / self.full_step, -1., 1.)
                index += 1
        # 一半随机候选改为当前计划附近的微调。
        start = max(index, self.candidate_count // 2)
        local[start:] = np.clip(base + .25 * local[start:], -1., 1.)
        local = self.bounded(view, local)
        proposals[:, view.owned] = local[:, view.owned]
        return proposals

    def flight_energy(self, actions):
        speeds = np.linalg.norm(actions * self.speed_scale, axis=-1)
        return self.propulsion(speeds) * self.flight_duration / self.energy_budget

    def cost(self, owner, view, proposals):
        prediction, uncertainty = self.models[owner].predict(view.features(proposals))
        # 本区域用户时延 + 飞行前由本 Master 管理的 UAV 能耗，避免迁移后重复计账。
        weights = np.concatenate(([self.delay_weight], view.queues * view.owned))
        fly = self.flight_energy(proposals) @ (view.queues * view.owned)
        return (np.maximum(prediction, 0.) @ weights + fly
                + self.uncertainty_penalty * (uncertainty @ weights))

    def actions(self, views, *, explore=0.):
        actions = np.zeros((self.count, 2))
        exploring = [rng.random() < explore for rng in self.rngs]
        evaluations = quote_messages = plan_messages = payload = 0
        for owner, (view, randomize) in enumerate(zip(views, exploring)):
            if randomize and view.owned.any():
                local = self.bounded(view, self.rngs[owner].uniform(-1., 1., (1, self.count, 2)))[0]
                actions[view.owned] = local[view.owned]
                plan_messages += len(self.neighbors[owner])
                payload += 2 * int(view.owned.sum()) * len(self.neighbors[owner])
        for _ in range(self.rounds):
            for owner, view in enumerate(views):
                if exploring[owner] or not view.owned.any():
                    continue
                proposals = self.candidates(view, actions, owner)
                scores = self.cost(owner, view, proposals)
                evaluations += len(proposals)
                for neighbor in self.neighbors[owner]:
                    # 请求只需发本 Master 管理 UAV 的计划；接收方按自己的可见范围报价。
                    scores += self.cost(neighbor, views[neighbor], proposals)
                    evaluations += len(proposals)
                    quote_messages += 2
                    payload += len(proposals) * (2 * int(view.owned.sum()) + 1)
                actions[view.owned] = proposals[int(np.argmin(scores)), view.owned]
                # 邻居后续报价所用的最新计划必须显式发布，不能把共享数组同步当作免费通信。
                plan_messages += len(self.neighbors[owner])
                payload += 2 * int(view.owned.sum()) * len(self.neighbors[owner])
        return actions, dict(candidate_evaluations=evaluations,
                             quote_messages=quote_messages, plan_messages=plan_messages,
                             neighbor_messages=quote_messages + plan_messages,
                             neighbor_payload_scalars=payload)
