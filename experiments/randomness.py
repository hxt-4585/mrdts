"""实验总 seed 的唯一随机流构建入口；组件只接收生成器。"""

import random

import numpy as np


class RandomStreams:
    """每回合独立的随机流；空间流固定，任务与策略流随回合变化。"""

    def __init__(self, seed, episode=0):
        spatial = np.random.SeedSequence([seed, 0]).generate_state(2)
        workload = int(np.random.SeedSequence([seed, episode]).generate_state(3)[2])
        self.region = np.random.default_rng(int(spatial[0]))
        self.user = np.random.default_rng(int(spatial[1]))
        self.dag = np.random.default_rng(workload)
        self.dag_python = random.Random(workload)
        self.flight = np.random.default_rng(np.random.SeedSequence([seed, episode, 101]))
        self.scheduling = np.random.default_rng(np.random.SeedSequence([seed, episode, 102]))

    @classmethod
    def from_config(cls, config=None, episode=0):
        """开发入口省略 config 时，也读取默认实验配置，不设备用 seed。"""
        if config is None:
            from experiments.config import load_config
            config = load_config()
        return cls(config.seed, episode)
