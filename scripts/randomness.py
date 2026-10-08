"""环境诊断和测试使用的随机流；默认 seed=42，不依赖方法配置。"""

import random

import numpy as np


class RandomStreams:
    """每回合独立的随机流；空间流固定，任务与策略流随回合变化。"""

    def __init__(self, seed=42, episode=0):
        spatial = np.random.SeedSequence([seed, 0]).generate_state(2)
        workload = int(np.random.SeedSequence([seed, episode]).generate_state(3)[2])
        self.region = np.random.default_rng(int(spatial[0]))
        self.user = np.random.default_rng(int(spatial[1]))
        self.dag = np.random.default_rng(workload)
        self.dag_python = random.Random(workload)
        self.flight = np.random.default_rng(np.random.SeedSequence([seed, episode, 101]))
        self.scheduling = np.random.default_rng(np.random.SeedSequence([seed, episode, 102]))
