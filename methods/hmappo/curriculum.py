"""低层预训练的外生飞行与场景分布；不修改环境动力学或奖励。"""

from dataclasses import replace

import numpy as np


MODES = ('hover', 'smooth', 'toward_users', 'away_users',
         'cluster', 'spread', 'boundary', 'cross_region')


def scenario_settings(base, episode, scene_seeds):
    """逐场景遍历全部飞行模式，交错负载/带宽；智能体数和观测维数固定。"""
    settings = dict(base)
    cycle = episode // len(MODES)
    seed = scene_seeds[cycle % len(scene_seeds)]
    load = (.5, 1., 1.5)[(episode + cycle) % 3]
    bandwidth = (.5, 1., 1.5)[(episode + 2 * cycle) % 3]
    minimum = base['region'].region_count * base['user'].min_users_per_region
    settings['env'] = replace(base['env'], env_seed=seed)
    settings['user'] = replace(base['user'], total_users=max(minimum, round(base['user'].total_users * load)))
    settings['dag'] = replace(base['dag'], n=max(2, round(base['dag'].n * load)))
    settings['channel'] = replace(base['channel'], total_bandwidth_mhz=base['channel'].total_bandwidth_mhz * bandwidth)
    return settings


class FlightCurriculum:
    """每 episode 固定模式，步间连续运动；所有动作满足速度及场地边界。

    起点涵盖原始部署、边缘、密集与分散布局。模式是采样覆盖而非穷举保证。
    """

    def __init__(self, env, episode, seed):
        self.mode = MODES[episode % len(MODES)]
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, env.env_config.env_seed, episode]))
        count = env.members.member_uav_count
        length = env.region.config.side_length
        self.velocity = np.zeros((count, 2))
        self.speed = self.rng.uniform(.1, 1., (count, 1))
        self.target = self.rng.uniform(.01, length - .01, (count, 2))
        positions = env.members.positions[:count, :2]
        layout = (episode + episode // len(MODES)) % 4
        if self.mode != 'hover':
            if layout == 1:
                positions[:] = self.target
                positions[np.arange(count), np.arange(count) % 2] = self.rng.choice([.01, length - .01], count)
            elif layout == 2:
                center = self.rng.uniform(.2 * length, .8 * length, 2)
                positions[:] = np.clip(center + self.rng.normal(0., .03 * length, (count, 2)), .01, length - .01)
            elif layout == 3:
                positions[:] = self.target
        env._refresh_topology()
        if self.mode == 'cluster':
            self.target[:] = env.users.positions[self.rng.integers(env.users.num_users), :2]
        elif self.mode == 'boundary':
            self.target[np.arange(count), np.arange(count) % 2] = self.rng.choice([.01, length - .01], count)
        elif self.mode == 'cross_region':
            self.target[:] = length - positions

    def actions(self, env):
        positions = env.members.positions[:env.members.member_uav_count, :2]
        length = env.region.config.side_length
        distance = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s
        if self.mode == 'hover':
            return np.zeros_like(positions, dtype=np.float32)
        if self.mode == 'smooth':
            self.velocity = .9 * self.velocity + .1 * self.rng.uniform(-1., 1., self.velocity.shape)
            direction = self.velocity
        elif self.mode in ('toward_users', 'away_users'):
            delta = env.users.positions[None, :, :2] - positions[:, None, :]
            nearest = np.argmin(np.square(delta).sum(-1), axis=1)
            direction = delta[np.arange(len(positions)), nearest]
            if self.mode == 'away_users':
                direction = -direction
        else:
            direction = self.target - positions
            if self.mode in ('spread', 'cross_region'):
                arrived = np.linalg.norm(direction, axis=1) < distance
                self.target[arrived] = self.rng.uniform(.01, length - .01, (int(arrived.sum()), 2))
                direction = self.target - positions
        action = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), distance)
        action *= self.speed
        # 裁剪的是生成的飞行动作，保持真实运动约束，不是对调用方输入作校验。
        lower = np.minimum(0., (.01 - positions) / distance)
        upper = np.maximum(0., (length - .01 - positions) / distance)
        return np.clip(action, lower, upper).astype(np.float32)
