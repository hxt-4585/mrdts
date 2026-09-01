"""最小 Environment 资源与链路能力的测试。"""

import unittest

import numpy as np

from env.channel_model import ChannelModel, LinkType
from env.channel_queue import EntityKind, EntityRef
from env.environment import Environment
from env.region import Region
from env.uav import MasterUAV, MemberUAV
from env.user import User


class TestEnvironment(unittest.TestCase):
    @staticmethod
    def _generate_members():
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()
        members.generate_from_region_and_user(region, users, masters)
        return members

    def test_lists_only_members_in_requested_region(self):
        members = self._generate_members()
        environment = Environment(members, ChannelModel(), user_transmit_power=0.1)
        region_id = int(members.region_ids[0])

        member_ids = environment.member_ids_in_region(region_id)

        expected_ids = np.flatnonzero(members.region_ids[: members.bs_index] == region_id)
        np.testing.assert_array_equal(member_ids, expected_ids)
        self.assertNotIn(members.bs_index, member_ids)
        self.assertEqual(environment.bs_index, members.bs_index)

    def test_estimate_does_not_reserve_but_reserve_updates_earliest_core(self):
        members = self._generate_members()
        environment = Environment(members, ChannelModel(), user_transmit_power=0.1)
        server_id = 0
        before = members.core_available_at.copy()

        start_time, estimated_finish = environment.estimate_finish_time(
            server_id, cpu_cycles=10e9, data_ready_time=0.5
        )

        np.testing.assert_array_equal(members.core_available_at, before)
        finish_time = environment.reserve_computation(server_id, cpu_cycles=10e9, data_ready_time=0.5)
        self.assertEqual(start_time, 0.5)
        self.assertEqual(finish_time, estimated_finish)
        self.assertEqual(members.core_available_at[server_id, 0], finish_time)

    def test_delegates_transmission_delay_to_channel_model(self):
        channel_model = ChannelModel()
        environment = Environment(self._generate_members(), channel_model, user_transmit_power=0.1)
        transmitter = np.array([0.0, 0.0, 0.0])
        receiver = np.array([0.0, 0.0, 50.0])

        delay = environment.transmission_delay_s(
            8_000.0,
            transmitter,
            receiver,
            transmit_power_w=0.1,
            link_type=LinkType.GROUND_TO_AIR,
        )

        self.assertEqual(
            delay,
            channel_model.transmission_delay_s(
                8_000.0,
                transmitter,
                receiver,
                transmit_power_w=0.1,
                link_type=LinkType.GROUND_TO_AIR,
            ),
        )

    def test_refresh_slot_topology_updates_member_regions_and_user_associations(self):
        region = Region()
        region.generate()
        users = User()
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()
        members.generate_from_region_and_user(region, users, masters)
        environment = Environment(members, ChannelModel(), user_transmit_power=0.1)
        initial_associations = np.array(
            [environment.member_ids_in_region(region_id)[0] for region_id in users.region_ids],
            dtype=int,
        )
        runtime = environment.create_scheduling_runtime(users.positions, initial_associations)
        target_region_id = int(next(region_id for region_id in range(1, 5)
                                    if region_id != members.region_ids[0]))
        target_row, target_column = np.argwhere(region.region_map == target_region_id)[0]
        members.positions[0, :2] = (
            (target_column + 0.5) * region.config.cell_size,
            (target_row + 0.5) * region.config.cell_size,
        )

        associations = environment.refresh_slot_topology(region, users.positions, runtime)

        self.assertEqual(members.region_ids[0], target_region_id)
        np.testing.assert_allclose(
            runtime.entity_positions[EntityRef(EntityKind.MEMBER_UAV, 0)],
            members.positions[0],
        )
        for user_id, member_id in enumerate(associations):
            self.assertEqual(
                members.region_ids[member_id],
                region.get_region_id(*users.positions[user_id, :2]),
            )
            self.assertEqual(
                runtime.ground_owner_members[EntityRef(EntityKind.GROUND_DEVICE, user_id)],
                EntityRef(EntityKind.MEMBER_UAV, int(member_id)),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
