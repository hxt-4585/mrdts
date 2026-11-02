"""随机基线通过正式环境接口运行，保留种子与结果格式。"""

from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from env.env import MRDTSEnv
from env.runtime.slot_result import DAGResult, SlotResult
from methods.random.run import (
    RandomStreams, build_scene, evaluate, resolved_settings, run_slot, slot_metrics, summarize,
)


class TestRandomRun(unittest.TestCase):
    def test_environment_seed_and_episode_control_scene_and_workload(self):
        settings = resolved_settings(users=8, dag_nodes=4)
        settings['env'] = replace(settings['env'], env_seed=123)
        first = build_scene(settings, episode=0)
        next_episode = build_scene(settings, episode=1)
        changed_settings = {**settings, 'env': replace(settings['env'], env_seed=124)}
        changed = build_scene(changed_settings, episode=0)

        np.testing.assert_array_equal(first.users.positions, next_episode.users.positions)
        np.testing.assert_array_equal(first.members.positions, next_episode.members.positions)
        self.assertFalse(np.array_equal(first.users.positions, changed.users.positions))
        first_dag = first.dag_generator.generate_single_dag()
        next_dag = next_episode.dag_generator.generate_single_dag()
        changed_dag = changed.dag_generator.generate_single_dag()
        self.assertFalse(np.array_equal(first_dag.node_features, next_dag.node_features))
        self.assertFalse(np.array_equal(first_dag.node_features, changed_dag.node_features))

    def test_method_randomness_does_not_change_scenes_or_episode_workloads(self):
        settings = resolved_settings(users=8, dag_nodes=4)
        settings['env'] = replace(settings['env'], env_seed=123)
        for episode in (0, 1):
            with self.subTest(episode=episode):
                streams = (RandomStreams(7, episode), RandomStreams(8, episode))
                scenes = [build_scene(settings, episode=episode) for _ in streams]
                streams[1].flight.random(1000)
                streams[1].scheduling.random(1000)
                first, second = scenes
                self.assertEqual(first.env_config.env_seed, 123)
                np.testing.assert_array_equal(first.region.region_map, second.region.region_map)
                np.testing.assert_array_equal(first.users.positions, second.users.positions)
                np.testing.assert_array_equal(first.members.positions, second.members.positions)
                np.testing.assert_array_equal(first.masters.positions, second.masters.positions)
                for _ in range(2):
                    workloads = []
                    for scene, stream in zip(scenes, streams):
                        generate = scene._generate_dag_requests

                        def capture():
                            requests = generate()
                            workloads.append(requests)
                            return requests

                        with patch.object(scene, '_generate_dag_requests', side_effect=capture):
                            run_slot(scene, stream)
                    for left, right in zip(*workloads):
                        self.assertEqual(left.ground_device, right.ground_device)
                        self.assertEqual(left.dag_id, right.dag_id)
                        self.assertEqual(left.dag.edges, right.dag.edges)
                        self.assertEqual(left.dag.edge_features, right.dag.edge_features)
                        np.testing.assert_array_equal(left.dag.node_features, right.dag.node_features)

    def test_metrics_include_actual_overtime_delays_and_weight_summary_by_dag_count(self):
        mixed = SlotResult(10., 15., (
            DAGResult((0, 0, 0), 10.5, False),
            DAGResult((1, 1, 0), 12., True),
            DAGResult((2, 2, 0), 15., True),
        ), 0., 0., ())
        overtime = SlotResult(15., 19., (DAGResult((0, 0, 1), 19., True),), 0., 0., ())
        first = slot_metrics(mixed, 0, 0, 0)
        second = slot_metrics(overtime, 0, 1, 1)

        self.assertEqual(first['delay_sum_s'], 7.5)
        self.assertEqual(first['mean_delay_s'], 2.5)
        self.assertEqual(first['successful_mean_delay_s'], .5)
        self.assertEqual(first['failure_rate'], 2 / 3)
        self.assertEqual(second['mean_delay_s'], 4.)
        self.assertIsNone(second['successful_mean_delay_s'])
        self.assertNotIn('truncated_delay_sum_s', first)
        self.assertNotIn('truncated_mean_delay_s', first)

        summary = summarize([first, second])
        self.assertEqual(summary['mean_delay_s'], 11.5 / 4)
        self.assertEqual(summary['successful_mean_delay_s'], .5)
        self.assertEqual(summary['failure_rate'], .75)
        self.assertEqual(summary['flight_violations'], 1)
        self.assertNotIn('truncated_mean_delay_s', summary)

    def test_reproducible_environment_runs_two_windows_with_energy_settlement(self):
        settings = resolved_settings(users=8, dag_nodes=4)
        first_streams, second_streams = RandomStreams(7, 2), RandomStreams(7, 2)
        first, second = build_scene(settings, episode=2), build_scene(settings, episode=2)
        self.assertIsInstance(first, MRDTSEnv)
        self.assertEqual(first.env_config.env_seed, settings['env'].env_seed)
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
        settings['env'] = replace(settings['env'], env_seed=123,
                                  steps_per_epoch=2, scheduling_duration_s=.001)
        with tempfile.TemporaryDirectory() as directory:
            with patch('methods.random.run.resolved_settings', return_value=settings), \
                    patch('builtins.print') as console:
                output = evaluate(seed=7, episodes=1, output_root=directory)
            messages = [call.args[0] for call in console.call_args_list]
            self.assertIn(str(output), messages[0])
            self.assertIn('环境种子 123', messages[1])
            self.assertIn('方法种子 7', messages[1])
            self.assertIn('1 episodes', messages[1])
            self.assertIn('2 slots/episode', messages[1])
            self.assertIn('Episode 1/1 | 初始化场景', messages[2])
            progress = [message for message in messages if ' | Slot ' in message]
            self.assertEqual(len(progress), 2)
            self.assertTrue(all(call.kwargs.get('flush') for call in console.call_args_list))
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['slots_completed'], 2)
            self.assertEqual(summary['dag_count'], 16)
            self.assertEqual(config['environment']['env']['env_seed'], 123)
            self.assertEqual(config['method_seed'], 7)
            metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['method_seed'], 7)
            self.assertEqual(metadata['env_seed'], 123)
            self.assertTrue((Path(output) / 'dags.csv').is_file())
            with (output / 'dags.csv').open(newline='', encoding='utf-8') as stream:
                dags = list(csv.DictReader(stream))
            with (output / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
                metrics = list(csv.DictReader(stream))
            self.assertTrue(any(dag['failed'] == 'True' for dag in dags))
            self.assertNotIn('truncated_delay_s', dags[0])
            slot_start = 0.
            for slot, metric in enumerate(metrics):
                self.assertIn(f'Episode 1/1 | Slot {slot + 1}/2', progress[slot])
                self.assertIn(f"平均时延 {float(metric['mean_delay_s']):.3f}s", progress[slot])
                self.assertIn(f"失败率 {float(metric['failure_rate']):.1%}", progress[slot])
                self.assertRegex(progress[slot], r'本步耗时 \d+\.\d{2}s \| 累计 \d+\.\d{1}s')
                current = [dag for dag in dags if int(dag['slot']) == slot]
                delays = [float(dag['delay_s']) for dag in current]
                for dag, delay in zip(current, delays):
                    self.assertAlmostEqual(delay, float(dag['completion_time_s']) - slot_start)
                self.assertAlmostEqual(float(metric['delay_sum_s']), sum(delays))
                self.assertAlmostEqual(float(metric['mean_delay_s']), sum(delays) / len(delays))
                slot_start = max(float(dag['completion_time_s']) for dag in current)
            self.assertAlmostEqual(summary['mean_delay_s'],
                                   sum(float(dag['delay_s']) for dag in dags) / len(dags))
