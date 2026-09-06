"""沿排序顺序，逐子任务从飞行后的合法执行节点中等概率选择。"""

import numpy as np

from methods.contracts import SchedulingContext


class RandomScheduling:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def schedule(self, context: SchedulingContext):
        placements = {}
        for key in context.order:
            candidates = context.candidates[key]
            if not candidates:
                raise ValueError(f"No legal execution node for {key}")
            placements[key] = candidates[int(self.rng.integers(len(candidates)))]
        return placements
