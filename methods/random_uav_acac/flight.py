"""Random flight specified by method seed, episode and resampling interval."""

import numpy as np


class RandomFlight:
    def __init__(self, env, seed, resample_slots):
        self.seed = int(np.random.SeedSequence([seed, 101]).generate_state(1)[0])
        self.interval = resample_slots
        self.count = env.members.member_uav_count
        self.distance = env.env_config.flight_duration_s * env.members.config.max_horizontal_speed
        self.edge = env.region.config.side_length
        self.initial_regions = env.members.region_ids[:self.count].copy()

    def actions(self, env, slot):
        if slot == 0:
            self.generator = np.random.default_rng(np.random.SeedSequence([self.seed, env.epoch_index]))
        if slot % self.interval == 0:
            self.vector = self.generator.uniform(-1, 1, (self.count, 2)).astype(np.float32)
        proposed = env.members.positions[:self.count, :2] + self.distance * self.vector
        self.vector[(proposed < 0) | (proposed > self.edge)] *= -1
        return self.vector

    def snapshot(self):
        return dict(policy='random_vector', flight_seed=self.seed, resample_slots=self.interval,
                    seed_components=['flight_seed', 'episode'], component_range=[-1., 1.],
                    boundary='reverse outward components', initial_region_ids=self.initial_regions.tolist())
