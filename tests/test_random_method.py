"""随机方案的复现、独立随机流和合法性验证。"""

from dataclasses import replace
import importlib
import unittest

import numpy as np

from experiments.config import load_config
from experiments.scene import build_scene, resolved_settings
from methods.contracts import FlightContext, SchedulingContext
from experiments.randomness import RandomStreams
from env.types import EntityKind, EntityRef, TaskKey
from methods.factory import create_method


class TestRandomMethod(unittest.TestCase):
    def test_default_is_ers_random_random(self):
        method = load_config().method
        self.assertEqual(method["solution"], "random")
        self.assertEqual(method["components"], dict(ordering="ers", flight="random", scheduling="random"))

    def test_random_flight_repeats_by_seed_and_changes_with_different_seed(self):
        try:
            flight = importlib.import_module("methods.components.flight.random").RandomFlight
        except ModuleNotFoundError:
            self.fail("Random flight component is missing")
        context = FlightContext(0, (3, 7), np.zeros((2, 3)))
        def actions(seed):
            return flight(RandomStreams(seed).flight).decide(context).member_actions
        seed = load_config().seed
        self.assertEqual(actions(seed), actions(seed))
        self.assertNotEqual(actions(seed), actions(seed + 1))
        values = np.asarray(list(actions(seed).values()))
        self.assertTrue(((values >= -1) & (values <= 1)).all())
        self.assertTrue(np.any(values != 0))

    def test_multiple_slots_preserve_service_and_complete_all_dags(self):
        cfg = replace(load_config(), users=4, dag_nodes=3)
        scene = build_scene(resolved_settings(cfg), randomness=RandomStreams.from_config(cfg, 0))
        method = create_method(cfg.method, flight_rng=RandomStreams.from_config(cfg, 0).flight, scheduling_rng=RandomStreams.from_config(cfg, 0).scheduling)
        positions = scene.simulator.members.positions.copy()
        for slot in range(20):
            outcome = method.run_slot(scene, scene.workload(slot))
            self.assertEqual(len(outcome.result.dags), 4)
            self.assertTrue(np.all(scene.simulator.user_member_ids >= 0))
        self.assertFalse(np.array_equal(positions, scene.simulator.members.positions))

    def test_random_scheduling_is_legal_repeatable_and_independent_of_flight_stream(self):
        cfg = load_config()
        first = create_method(cfg.method, flight_rng=RandomStreams.from_config(cfg, 0).flight, scheduling_rng=RandomStreams.from_config(cfg, 0).scheduling)
        second = create_method(cfg.method, flight_rng=RandomStreams.from_config(cfg, 0).flight, scheduling_rng=RandomStreams.from_config(cfg, 0).scheduling)
        owner = EntityRef(EntityKind.MEMBER_UAV, 0)
        ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        keys = tuple(TaskKey(0, 0, 0, i) for i in range(40))
        context = SchedulingContext(owner, (), keys, {key: (ground, owner) for key in keys})
        selected = first.scheduling.schedule(context)
        self.assertEqual(selected, second.scheduling.schedule(context))
        self.assertEqual(set(selected), set(keys))
        self.assertEqual(set(selected.values()), {ground, owner})
        first.scheduling.schedule(context)
        flight_context = FlightContext(0, (0,), np.zeros((1, 3)))
        self.assertEqual(first.flight.decide(flight_context), second.flight.decide(flight_context))

    def test_service_region_rejections_fall_back_without_mutating_preview_state(self):
        cfg = replace(load_config(), users=4, dag_nodes=3)
        scene = build_scene(resolved_settings(cfg), randomness=RandomStreams.from_config(cfg, 0))
        method = create_method(cfg.method, flight_rng=RandomStreams.from_config(cfg, 0).flight, scheduling_rng=RandomStreams.from_config(cfg, 0).scheduling)
        members = scene.simulator.members
        members.config = replace(members.config, max_horizontal_speed=scene.region.config.side_length,
                                 flight_duration=1.0)
        before = members.positions.copy()
        # 所有 Member 飞向同一区域，必然使另三个有用户区域失去服务。
        actions = (scene.users.positions[0, :2] - before[:members.bs_index, :2]) / scene.region.config.side_length
        method.flight_actions = lambda _: actions
        method.max_flight_attempts = 2
        outcome = method.run_slot(scene, scene.workload(0))
        np.testing.assert_array_equal(before, members.positions)
        self.assertEqual(outcome.flight_resamples, 2)
        self.assertEqual(outcome.flight_fallbacks, 1)
        self.assertEqual(len(outcome.result.dags), 4)


if __name__ == "__main__":
    unittest.main()
