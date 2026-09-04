"""原始输入中继与前驱结果路由测试。"""

import unittest

from env.communication.routing import RoutePlanner
from env.types import DirectedChannelKey, EntityKind, EntityRef


class TestRoutePlanner(unittest.TestCase):
    def setUp(self):
        self.ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        self.owner = EntityRef(EntityKind.MEMBER_UAV, 1)
        self.other_member = EntityRef(EntityKind.MEMBER_UAV, 2)
        self.bs = EntityRef(EntityKind.BS, 0)
        self.planner = RoutePlanner()

    def test_input_to_bs_uses_owner_member_as_relay(self):
        route = self.planner.input_route(self.ground, self.owner, self.bs)
        self.assertEqual(
            route.hops,
            (
                DirectedChannelKey(self.ground, self.owner),
                DirectedChannelKey(self.owner, self.bs),
            ),
        )

    def test_input_to_other_member_uses_owner_member_as_relay(self):
        route = self.planner.input_route(self.ground, self.owner, self.other_member)
        self.assertEqual(
            route.hops,
            (
                DirectedChannelKey(self.ground, self.owner),
                DirectedChannelKey(self.owner, self.other_member),
            ),
        )

    def test_input_executed_on_source_ground_is_local(self):
        route = self.planner.input_route(self.ground, self.owner, self.ground)

        self.assertEqual(route.hops, ())

    def test_result_from_ground_to_bs_uses_owner_member_as_relay(self):
        route = self.planner.predecessor_route(
            self.ground,
            self.bs,
            ground_device=self.ground,
            owner_member=self.owner,
        )

        self.assertEqual(
            route.hops,
            (
                DirectedChannelKey(self.ground, self.owner),
                DirectedChannelKey(self.owner, self.bs),
            ),
        )

    def test_result_from_bs_to_ground_uses_owner_member_as_relay(self):
        route = self.planner.predecessor_route(
            self.bs,
            self.ground,
            ground_device=self.ground,
            owner_member=self.owner,
        )

        self.assertEqual(
            route.hops,
            (
                DirectedChannelKey(self.bs, self.owner),
                DirectedChannelKey(self.owner, self.ground),
            ),
        )

    def test_result_between_same_execution_node_is_local(self):
        self.assertEqual(
            self.planner.predecessor_route(
                self.bs,
                self.bs,
                ground_device=self.ground,
                owner_member=self.owner,
            ).hops,
            (),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
