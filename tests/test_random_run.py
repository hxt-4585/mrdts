"""随机基线通过正式环境接口运行，保留种子与结果格式。"""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from env.env import MRDTSEnv
from methods.random.run import RandomStreams, build_scene, evaluate, resolved_settings, run_slot


class TestRandomRun(unittest.TestCase):
    def test_reproducible_environment_runs_two_windows_with_energy_settlement(self):
        settings = resolved_settings(users=8, dag_nodes=4)
        first_streams, second_streams = RandomStreams(7, 2), RandomStreams(7, 2)
        first, second = build_scene(settings, first_streams), build_scene(settings, second_streams)
        self.assertIsInstance(first, MRDTSEnv)
        self.assertEqual(first.env_config.env_seed, 7)
        self.assertEqual(first.epoch_index, 2)
        for step in range(2):
            result, violations = run_slot(first, first_streams)
            replay, replay_violations = run_slot(second, second_streams)
            self.assertEqual(result, replay)
            self.assertEqual(violations, replay_violations)
            self.assertEqual(first.step_index, step + 1)
            self.assertIsNone(first._runtime)
            self.assertEqual(result.slot_start, step * settings['env'].scheduling_duration_s)
            self.assertEqual([dag.key[2] for dag in sorted(result.dags, key=lambda d: d.key[1])],
                             list(range(step * 8, (step + 1) * 8)))
            np.testing.assert_array_equal(first.members.positions, second.members.positions)
            np.testing.assert_array_equal(first.members.energy_virtual_queue_j, second.members.energy_virtual_queue_j)

    def test_evaluate_retains_outputs_and_records_actual_environment_seed(self):
        settings = resolved_settings(users=8, dag_nodes=4)
        settings['env'] = replace(settings['env'], steps_per_epoch=2)
        with tempfile.TemporaryDirectory() as directory:
            with patch('methods.random.run.resolved_settings', return_value=settings):
                output = evaluate(seed=7, episodes=1, output_root=directory)
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['slots_completed'], 2)
            self.assertEqual(summary['dag_count'], 16)
            self.assertEqual(config['environment']['env']['env_seed'], 7)
            self.assertTrue((Path(output) / 'dags.csv').is_file())
