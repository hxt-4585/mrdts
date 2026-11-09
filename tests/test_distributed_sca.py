"""Non-neural, locally observable trajectory optimization contracts."""

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class TestAnalyticPlanner(unittest.TestCase):
    def planner_module(self):
        self.assertIsNotNone(importlib.util.find_spec("methods.distributed_sca"),
                             "independent SCA baseline is not implemented")
        from methods.distributed_sca import planner
        return planner

    def problem(self, points, sources, targets, bits, airborne, owned=(0,), ids=(0,)):
        p = self.planner_module()
        from env.settings import ChannelConfig
        return p.LocalProblem(
            np.array(ids), np.array(owned), np.array(points, dtype=float),
            np.array(sources), np.array(targets), np.array(bits, dtype=float),
            np.ones(len(bits)), np.array(airborne), ChannelConfig.default(), 0.1)

    def test_delay_and_gradient_match_channel_physics(self):
        from env.communication.channel_model import ChannelModel, LinkType
        from env.settings import ChannelConfig
        problem = self.problem([[170., 40., 50.], [0., 0., 0.]], [1], [0], [1e6], [False])
        cost, gradient = problem.evaluate()
        expected = ChannelModel(ChannelConfig.default()).transmission_delay_s(
            1e6, problem.positions[1], problem.positions[0], 1., LinkType.GROUND_TO_AIR)
        self.assertAlmostEqual(cost, expected, delta=expected * 1e-5)
        for axis in range(2):
            original = problem.positions[0, axis]
            problem.positions[0, axis] = original + 1e-3
            plus = problem.evaluate()[0]
            problem.positions[0, axis] = original - 1e-3
            minus = problem.evaluate()[0]
            problem.positions[0, axis] = original
            self.assertAlmostEqual(gradient[0, axis], (plus - minus) / .002, delta=1e-10)

    def test_air_to_air_gradient_has_equal_opposite_endpoints(self):
        problem = self.problem([[100., 20., 50.], [200., 80., 50.]], [0], [1], [2e6], [True],
                               owned=(0, 1), ids=(0, 1))
        _, gradient = problem.evaluate()
        np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-15)
        self.assertLess(gradient[0, 0], 0.)

    def test_sca_decreases_proxy_and_respects_per_slot_flight_box(self):
        p = self.planner_module()
        problem = self.problem([[100., 100., 50.], [220., 160., 0.]], [1], [0], [1e6], [False])
        before = problem.evaluate()[0]
        action, metrics = p.coordinate([problem], [()], 5., 1000., rounds=3, backtracks=6)
        self.assertLess(problem.evaluate()[0], before)
        self.assertGreater(action[0, 0], 0.)
        self.assertTrue((np.abs(action) <= 1.).all())
        self.assertEqual(metrics["block_attempts"], 3)
        self.assertLessEqual(metrics["oracle_evaluations"], 3 * 7)
        self.assertGreater(metrics["accepted_steps"], 0)

    def test_neighbor_cost_gradient_changes_the_owners_action(self):
        p = self.planner_module()
        own = self.problem([[150., 100., 50.], [350., 100., 50.]], [], [], [], [],
                           owned=(0,), ids=(0, 1))
        neighbor = self.problem([[150., 100., 50.], [350., 100., 50.]], [0], [1], [1e6], [True],
                                owned=(1,), ids=(0, 1))
        action, metrics = p.coordinate([own, neighbor], [(1,), (0,)], 5., 1000., 3, 6)
        self.assertGreater(action[0, 0], 0.)
        self.assertLess(action[1, 0], 0.)
        np.testing.assert_allclose(own.positions[:2], neighbor.positions[:2])
        self.assertGreater(metrics["neighbor_oracle_calls"], 0)

    def test_projected_actions_remain_valid_in_float32_environment_arithmetic(self):
        p = self.planner_module()
        origin = np.array([[999.9, .1]], dtype=np.float32)
        problem = self.problem([[float(origin[0, 0]), float(origin[0, 1]), 50.], [1000., 0., 0.]],
                               [1], [0], [1e6], [False])
        action, _ = p.coordinate([problem], [()], 5., 1000., 3, 6)
        physical = origin + action.astype(np.float32) * np.float32(10.) * np.float32(.5)
        self.assertTrue(((physical >= 0.) & (physical <= 1000.)).all())

    def test_planner_has_no_torch_or_other_baseline_dependencies(self):
        p = self.planner_module()
        tree = ast.parse(Path(p.__file__).read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names]
        self.assertFalse(any(name and (name.startswith("torch") or name.startswith("methods."))
                             for name in imports))


class TestIndependentExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        checkpoint = ROOT / "results/random_uav_ppo/seed_45/train_20261103T162321_743467Z_84a29361/checkpoint.pt"
        source = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cls.source = {"environment": source["environment"], "member_policies": source["member_policies"]}

    def engine_module(self):
        self.assertIsNotNone(importlib.util.find_spec("methods.distributed_sca.engine"),
                             "independent execution and local measurement adapter is missing")
        from methods.distributed_sca import engine
        return engine

    def test_checkpoint_scene_is_restored_and_low_policy_is_frozen(self):
        engine = self.engine_module()
        sim = engine.Simulator(self.source, slots=2)
        self.assertEqual(sim.env.users.num_users, 30)
        self.assertEqual(sim.env.settings["channel"].total_bandwidth_mhz, 80.)
        before = [tensor.clone() for actor in sim.actors for layer in actor for tensor in layer]
        sim.env.reset(epoch_index=2500000)
        result, traffic = sim.step(np.zeros((12, 2)))
        self.assertEqual(len(result.slot_result.dags), 30)
        self.assertFalse(any(dag.failed for dag in result.slot_result.dags))
        self.assertTrue(any(traffic))
        for old, new in zip(before, [t for actor in sim.actors for layer in actor for t in layer]):
            self.assertFalse(new.requires_grad)
            np.testing.assert_array_equal(old.numpy(), new.numpy())

    def test_only_local_users_and_one_hop_member_positions_enter_problem(self):
        engine = self.engine_module()
        sim = engine.Simulator(self.source, slots=2)
        env = sim.env
        master = next(i for i in range(4) if len(env.masters.neighbor_region_ids[i + 1]) < 3)
        first = engine.build_problem(env, master, None, .15, .1)
        local_users = np.flatnonzero(env.users.region_ids == master + 1)
        self.assertEqual(len(first.positions), len(first.member_ids) + len(local_users) + 1)
        hidden = np.setdiff1d(np.arange(12), first.member_ids)
        env.members.positions[hidden, :2] = 1e6
        env.members.energy_virtual_queue_j[:] = 1e10
        second = engine.build_problem(env, master, None, .15, .1)
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.weighted_bits, second.weighted_bits)

    def test_recorded_traffic_is_local_and_not_a_future_workload_query(self):
        engine = self.engine_module()
        from env.types import EntityKind
        sim = engine.Simulator(self.source, slots=2)
        sim.env.reset(epoch_index=2500000)
        _, traffic = sim.step(np.zeros((12, 2)))
        for master, links in enumerate(traffic):
            for (source, target), bits in links.items():
                self.assertGreater(bits, 0.)
                for endpoint in (source, target):
                    if endpoint.kind is EntityKind.GROUND_DEVICE:
                        self.assertEqual(sim.env.users.region_ids[endpoint.index], master + 1)
                    elif endpoint.kind is EntityKind.MEMBER_UAV:
                        self.assertEqual(sim.env.members.region_ids[endpoint.index], master + 1)
        rng_before = repr(sim.env.dag_generator._np_rng.bit_generator.state)
        for master in range(4):
            engine.build_problem(sim.env, master, traffic[master], .15, .1)
        self.assertEqual(repr(sim.env.dag_generator._np_rng.bit_generator.state), rng_before)

    def test_independent_ers_matches_manual_core_weighted_rank_order(self):
        engine = self.engine_module()
        from env.communication.channel_model import ChannelModel
        from env.contracts import DAGRequest
        from env.runtime.event_runtime import SchedulingRuntime, ServerSpec
        from env.types import EntityKind, EntityRef
        from env.workload.dag_generator import DAG
        ground, member, bs = (EntityRef(kind, 0) for kind in
                              (EntityKind.GROUND_DEVICE, EntityKind.MEMBER_UAV, EntityKind.BS))
        runtime = SchedulingRuntime(ChannelModel(),
            {ground: np.array([0., 0., 0.]), member: np.array([0., 0., 100.]), bs: np.array([100., 0., 20.])},
            {ground: ServerSpec((1e9,), 0.), member: ServerSpec((10e9,) * 2, 0.), bs: ServerSpec((12e9,) * 4, 0.)},
            {ground: 1., member: 1., bs: 1.}, member_regions={member: 1}, ground_owner_members={ground: member})
        requests = [DAGRequest(d, DAG(2, [(0, 1)], np.array([[1000., c] for c in cycles]), {(0, 1): 1000.}), member, ground)
                    for d, cycles in enumerate(((4e8, 1e8), (2e8, 2e8)))]
        orders = engine.ranked_orders(runtime, requests)
        self.assertEqual([(key.dag_id, key.node_id) for key in orders[0]], [(0, 0), (1, 0), (1, 1), (0, 1)])

    def test_all_new_runtime_imports_are_independent(self):
        self.engine_module()
        for file in (ROOT / "methods/distributed_sca").glob("*.py"):
            tree = ast.parse(file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("methods."):
                    self.assertTrue(node.module.startswith("methods.distributed_sca"), file.name)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith("methods.") and not alias.name.startswith("methods.distributed_sca"))


class TestEvaluation(unittest.TestCase):
    def runner(self):
        self.assertIsNotNone(importlib.util.find_spec("methods.distributed_sca.run"),
                             "paired evaluation entry point is missing")
        from methods.distributed_sca import run
        return run

    def test_paired_report_uses_episodes_and_labels_inferiority(self):
        run = self.runner()
        records = [dict(epoch=i, sca=dict(mean_delay_s=.23 + i * .001),
                        cem=dict(mean_delay_s=.21 + i * .001)) for i in range(4)]
        report = run.compare(records, seed=45)
        self.assertAlmostEqual(report["difference_s"], .02)
        self.assertGreater(report["paired_difference_ci95_s"][0], 0.)
        self.assertFalse(report["within_1pct"])
        self.assertFalse(report["ci_supports_1pct"])

    def test_short_end_to_end_is_repeatable_and_does_not_train(self):
        run = self.runner()
        config = run.load_config(ROOT / "config/methods/distributed_sca.toml")
        config.update(slots=2, epochs=[2500000], workers=1)
        with tempfile.TemporaryDirectory(prefix="mrdts_sca_") as directory:
            report = run.evaluate(config, Path(directory))
            saved = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(report, saved)
            self.assertTrue(report["low_actor_identical"])
            self.assertEqual(report["episodes"][0]["sca"]["failed_dags"], 0)
            self.assertEqual(report["episodes"][0]["sca"]["boundary_violations"], 0)
            self.assertLessEqual(report["episodes"][0]["sca"]["max_proxy_increase_s"], 1e-12)
            self.assertEqual(len(list(Path(directory).glob("*.pt"))), 0)
        source, target = run.load_inputs(config)
        first = run.run_episode(source, config, 2500000, None)
        second = run.run_episode(source, config, 2500000, None)
        self.assertEqual(first["slot_delay_s"], second["slot_delay_s"])
        self.assertEqual(first["final_positions_m"], second["final_positions_m"])

    def test_monitor_distinguishes_completion_and_unexpected_exit(self):
        self.assertIsNotNone(importlib.util.find_spec("methods.distributed_sca.monitor"),
                             "background evaluation monitor is missing")
        from methods.distributed_sca.monitor import snapshot
        with tempfile.TemporaryDirectory(prefix="mrdts_sca_monitor_") as directory:
            folder = Path(directory)
            self.assertEqual(snapshot(folder, True)["status"], "starting")
            self.assertEqual(snapshot(folder, False)["status"], "failed")
            (folder / "progress.json").write_text(json.dumps(dict(status="completed", pairs_completed=10)))
            self.assertEqual(snapshot(folder, True)["status"], "completed")

    def test_real_multiprocess_entry_point(self):
        run = self.runner()
        import subprocess
        import sys
        with tempfile.TemporaryDirectory(prefix="mrdts_sca_spawn_") as directory:
            result = subprocess.run([sys.executable, "-B", "-m", "methods.distributed_sca.run",
                                     "--slots", "1", "--epochs", "2500000", "2500001",
                                     "--workers", "2", "--output", directory], cwd=ROOT,
                                    text=True, capture_output=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
            self.assertEqual([item["epoch"] for item in report["episodes"]], [2500000, 2500001])
            self.assertTrue(report["reference_environment_identical"])


if __name__ == "__main__":
    unittest.main()
