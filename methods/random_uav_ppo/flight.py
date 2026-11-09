"""定期独立采样二维动作；外边界处反转将越界的分量。"""

import numpy as np


def derive_flight_seed(seed):
    """由方法种子派生飞行子流，隔离网络和调度采样的随机数消耗。"""
    return int(np.random.SeedSequence([seed, 101]).generate_state(1)[0])


class RandomFlight:
    def __init__(self, env, flight_seed=46, resample_slots=10):
        self.count = env.members.member_uav_count
        self.regions = env.members.region_ids[:self.count].copy()
        self.flight_seed = flight_seed
        self.resample_slots = resample_slots
        self.full_step = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s
        self.side_length = env.region.config.side_length

    def actions(self, env, slot):
        if slot == 0:
            # 独立飞行随机流；重置相同 episode 可重放，策略采样不会改变轨迹。
            self.rng = np.random.default_rng(np.random.SeedSequence([self.flight_seed, env.epoch_index]))
        if slot % self.resample_slots == 0:
            self.current_actions = self.rng.uniform(-1., 1., size=(self.count, 2)).astype(np.float32)
        actions = self.current_actions
        target = env.members.positions[:self.count, :2] + actions * self.full_step
        outward = (target < 0.) | (target > self.side_length)
        actions[outward] *= -1.
        return actions

    def snapshot(self):
        return dict(policy='random_vector', repeat_each_episode=False,
                    flight_seed=self.flight_seed, resample_slots=self.resample_slots, seed_components=['flight_seed', 'episode'],
                    action_components=['a_x', 'a_y'], component_range=[-1., 1.],
                    sampling='independent uniform components for every UAV at each resampling interval',
                    boundary='reverse each component whose proposed endpoint leaves the map',
                    initial_region_ids=self.regions.tolist())
