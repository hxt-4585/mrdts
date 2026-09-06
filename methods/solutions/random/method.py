"""随机方案；按现有物理规则预览飞行，拒绝迁空服务区域的联合提议。"""

from copy import copy

import numpy as np

from methods.compose import CompositeMethod


class RandomMethod(CompositeMethod):
    max_flight_attempts = 128

    def begin_slot(self, scene):
        simulator = scene.simulator
        if simulator.runtime is not None:
            raise RuntimeError("Previous slot has not ended")
        required_regions = set(int(region_id) for region_id in scene.users.region_ids)
        self.flight_resamples = 0
        self.flight_fallbacks = 0
        for attempt in range(self.max_flight_attempts):
            actions = self.flight_actions(scene)
            # 复用真实移动规则，仅复制位置；预览不改变真实位置、归属或时钟。
            preview = copy(simulator.members)
            preview.positions = simulator.members.positions.copy()
            preview.apply_flight_actions(actions, scene.region.config.side_length)
            occupied = {scene.region.get_region_id(*position[:2])
                        for position in preview.positions[:preview.bs_index]}
            if required_regions <= occupied:
                return simulator.begin_slot(scene.region, scene.users.positions, actions)
            self.flight_resamples += 1
        self.flight_fallbacks = 1
        return simulator.begin_slot(scene.region, scene.users.positions,
                                    np.zeros((simulator.members.member_uav_count, 2)))


def build_method(ordering, flight, scheduling):
    return RandomMethod(ordering, flight, scheduling)
