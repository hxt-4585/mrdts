"""正式实验组合、复现与入口的行为验证。"""

from experiments.randomness import RandomStreams
from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class TestExperiments(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module("experiments.config"), importlib.import_module("experiments.runner")
        except ModuleNotFoundError as exc:
            self.fail(f"正式实验入口尚未实现: {exc}")

    def test_repeatable_metrics_and_unique_output_directories(self):
        config_module, runner = self.api()
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(config_module.load_config(), slots=2, episodes=2,
                          output_root=Path(directory), users=4, dag_nodes=3)
            first = runner.evaluate(cfg)
            second = runner.evaluate(cfg)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "metrics.csv").read_text(), (second / "metrics.csv").read_text())
            metadata = json.loads((first / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertTrue((first / "config.json").is_file())
            snapshot = json.loads((first / "config.json").read_text())
            self.assertEqual(snapshot["experiment"]["seed"], cfg.seed)
            for scene in snapshot["scenes"]:
                for module_config in scene.values():
                    self.assertNotIn("seed", module_config)
            summary = json.loads((first / "summary.json").read_text())
            self.assertEqual(summary["dag_count"], 16)

    def test_fixed_positions_across_episodes_with_fresh_repeatable_dags(self):
        import numpy as np
        from experiments.config import load_config
        from experiments.scene import build_scene, resolved_settings
        from methods.factory import create_method

        cfg = replace(load_config(), users=4, dag_nodes=3)
        first = build_scene(resolved_settings(cfg), randomness=RandomStreams.from_config(cfg, 0))
        second = build_scene(resolved_settings(cfg), randomness=RandomStreams.from_config(cfg, 1))
        replay = build_scene(resolved_settings(cfg), randomness=RandomStreams.from_config(cfg, 1))
        for left, right in ((first, second), (second, replay)):
            np.testing.assert_array_equal(left.region.region_map, right.region.region_map)
            np.testing.assert_array_equal(left.users.positions, right.users.positions)
            np.testing.assert_array_equal(left.masters.positions, right.masters.positions)
            np.testing.assert_array_equal(left.simulator.members.positions, right.simulator.members.positions)

        def signature(workload):
            return [(user, dag.edges, dag.node_features.tolist(), dag.edge_features)
                    for _, user, dag in workload]

        first_tasks = first.workload(0)
        second_tasks = second.workload(0)
        self.assertNotEqual(signature(first_tasks), signature(second_tasks))
        self.assertEqual(signature(second_tasks), signature(replay.workload(0)))
        next_tasks = second.workload(1)
        self.assertNotEqual(signature(second_tasks), signature(next_tasks))
        self.assertEqual(signature(next_tasks), signature(replay.workload(1)))

        initial_users = first.users.positions.copy()
        initial_members = first.simulator.members.positions.copy()
        method = create_method(cfg.method, flight_rng=RandomStreams.from_config(cfg, 0).flight, scheduling_rng=RandomStreams.from_config(cfg, 0).scheduling)
        method.run_slot(first, first_tasks)
        np.testing.assert_array_equal(first.users.positions, initial_users)
        self.assertFalse(np.array_equal(first.simulator.members.positions, initial_members))
        reset = build_scene(resolved_settings(cfg), randomness=RandomStreams.from_config(cfg, 2))
        np.testing.assert_array_equal(reset.users.positions, initial_users)
        np.testing.assert_array_equal(reset.simulator.members.positions, initial_members)

        changed_config = replace(cfg, seed=cfg.seed + 1)
        changed = build_scene(resolved_settings(changed_config),
                              randomness=RandomStreams.from_config(changed_config))
        self.assertFalse(np.array_equal(changed.users.positions, initial_users))
        self.assertNotEqual(signature(changed.workload(0)), signature(first_tasks))

    def test_swappable_scheduler_produces_complete_placements(self):
        config_module, runner = self.api()
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(config_module.load_config(), slots=1, episodes=1,
                          output_root=Path(directory), users=4, dag_nodes=3)
            for scheduler in ("local", "owner"):
                with self.subTest(scheduler=scheduler):
                    method = {**cfg.method, "components": {**cfg.method["components"], "scheduling": scheduler}}
                    path = runner.evaluate(replace(cfg, method=method))
                    result = json.loads((path / "summary.json").read_text())
                    self.assertEqual(result["dag_count"], 4)

    def test_direct_entry_works_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run([sys.executable, str(ROOT / "experiments" / "run.py"),
                                      "--episodes", "1", "--slots", "1", "--output", directory],
                                     cwd=directory, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(len(list(Path(directory).rglob("summary.json"))), 1)

    def test_missing_trainer_is_explicit_and_does_not_run_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run([sys.executable, str(ROOT / "experiments" / "train.py"),
                                      "--device", "cpu", "--output", directory],
                                     cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("trainer", process.stderr.lower())
            self.assertEqual(list(Path(directory).rglob("metrics.csv")), [])

    def test_registered_trainer_receives_config_and_selected_device(self):
        from experiments.train import main
        received = []

        class RecordingTrainer:
            def train(self, config, device):
                received.append((config.seed, config.slots, device))
                return "trainer-result"

        with patch.dict("methods.factory.TRAINERS", {"random": RecordingTrainer}), \
                patch("experiments.train.check_device", return_value=("cuda:0", "test device")):
            result = main(["--seed", "73", "--slots", "2", "--device", "cuda:0"])
        self.assertEqual(result, "trainer-result")
        self.assertEqual(received, [(73, 2, "cuda:0")])

    def test_incomplete_member_decision_is_rejected_before_any_submission(self):
        from experiments.config import load_config
        from experiments.scene import build_scene, resolved_settings
        from methods.factory import create_method
        cfg = replace(load_config(), users=4, dag_nodes=3)
        scene = build_scene(resolved_settings(cfg), randomness=RandomStreams.from_config(cfg, 0))
        method = create_method(cfg.method, flight_rng=RandomStreams.from_config(cfg, 0).flight, scheduling_rng=RandomStreams.from_config(cfg, 0).scheduling)

        class MissingPlacements:
            def schedule(self, context):
                return {}

        method.scheduling = MissingPlacements()
        with self.assertRaisesRegex(ValueError, "every owned task"):
            method.run_slot(scene, scene.workload(0))
        self.assertEqual(scene.simulator.runtime.tasks, {})

    def test_all_failed_dags_remain_in_delay_denominator(self):
        from env.runtime.slot_result import DAGResult, SlotResult
        from experiments.metrics import slot_metrics, summarize
        from methods.compose import SlotOutcome
        result = SlotResult(2.0, 3.0, (DAGResult((0, 0, 0), None, True),), 0.0, 0.0)
        summary = summarize([slot_metrics(SlotOutcome(result, 0), 0, 0)])
        self.assertEqual(summary["failure_rate"], 1.0)
        self.assertEqual(summary["truncated_mean_delay_s"], 1.0)
        self.assertIsNone(summary["successful_mean_delay_s"])


if __name__ == "__main__":
    unittest.main()
