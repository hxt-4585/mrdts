"""Master 对所属 Member 的两个归一化动作分量独立均匀采样。"""

import numpy as np

from methods.contracts import FlightContext, FlightDecision


class RandomFlight:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def decide(self, context: FlightContext) -> FlightDecision:
        values = self.rng.uniform(-1.0, 1.0, size=(len(context.member_ids), 2))
        return FlightDecision({member_id: tuple(float(v) for v in action)
                               for member_id, action in zip(context.member_ids, values)})
