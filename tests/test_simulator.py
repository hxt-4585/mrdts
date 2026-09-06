"""最小 Simulator 资源与链路能力的测试。"""

import unittest

import numpy as np

from experiments.randomness import RandomStreams
from env.communication.channel_model import ChannelModel, LinkType
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User
from env.simulator import Simulator
from env.types import EntityKind, EntityRef


class TestSimulator(unittest.TestCase):
    @staticmethod
    def _generate_members():
        region = Region(rng=RandomStreams.from_config().region)
        region.generate()
        users = User(rng=RandomStreams.from_config().user)
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()
        members.generate_from_region_and_user(region, users, masters)
        return members

    def test_lists_only_members_in_requested_region(self):
        members = self._generate_members()
        environment = Simulator(members, ChannelModel(), user_transmit_power=0.1)
        region_id = int(members.region_ids[0])

        member_ids = environment.member_ids_in_region(region_id)

        expected_ids = np.flatnonzero(members.region_ids[: members.bs_index] == region_id)
        np.testing.assert_array_equal(member_ids, expected_ids)
        self.assertNotIn(members.bs_index, member_ids)
        self.assertEqual(environment.bs_index, members.bs_index)

    def test_does_not_expose_a_second_compute_reservation_state(self):
        """Simulator 不应在 SchedulingRuntime 之外维护计算预留状态。"""
        members = self._generate_members()
        environment = Simulator(members, ChannelModel(), user_transmit_power=0.1)

        self.assertFalse(hasattr(environment, "estimate_finish_time"))
        self.assertFalse(hasattr(environment, "reserve_computation"))

    def test_delegates_transmission_delay_to_channel_model(self):
        channel_model = ChannelModel()
        environment = Simulator(self._generate_members(), channel_model, user_transmit_power=0.1)
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

    def test_registers_each_ground_device_as_a_single_core_zero_energy_server(self):
        members = self._generate_members()
        environment = Simulator(
            members,
            ChannelModel(),
            user_transmit_power=0.1,
            user_core_frequency=1e9,
        )
        user_positions = np.array(
            [[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=float
        )
        runtime = environment.create_scheduling_runtime(
            user_positions,
            np.array([0, 0], dtype=int),
        )

        for user_id in range(len(user_positions)):
            server = runtime.servers[EntityRef(EntityKind.GROUND_DEVICE, user_id)]
            self.assertEqual(server.core_frequencies, (1e9,))
            self.assertEqual(server.capacitance_factor, 0.0)

    def test_rejects_non_positive_ground_core_frequency(self):
        with self.assertRaisesRegex(ValueError, "user_core_frequency"):
            Simulator(
                self._generate_members(),
                ChannelModel(),
                user_transmit_power=0.1,
                user_core_frequency=0.0,
            )

    def test_refresh_slot_topology_updates_member_regions_and_user_associations(self):
        region = Region(rng=RandomStreams.from_config().region)
        region.generate()
        users = User(rng=RandomStreams.from_config().user)
        users.generate_from_region(region)
        masters = MasterUAV()
        masters.generate_from_region(region)
        members = MemberUAV()
        members.generate_from_region_and_user(region, users, masters)
        environment = Simulator(members, ChannelModel(), user_transmit_power=0.1)
        target_region_id = int(next(region_id for region_id in range(1, 5)
                                    if region_id != members.region_ids[0]))
        target_row, target_column = np.argwhere(region.region_map == target_region_id)[0]
        members.positions[0, :2] = (
            (target_column + 0.5) * region.config.cell_size,
            (target_row + 0.5) * region.config.cell_size,
        )

        associations = environment.refresh_slot_topology(region, users.positions)
        runtime = environment.create_scheduling_runtime(users.positions, associations)

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
