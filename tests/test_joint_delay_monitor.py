import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class JointMonitorTests(unittest.TestCase):
    def test_terminal_runs_produce_paired_comparison_and_completion(self):
        from scripts.monitor_joint_delay import inspect
        with tempfile.TemporaryDirectory(prefix='joint_monitor_') as temporary:
            root = Path(temporary)
            runs = {}
            for name, delay in (('joint_delay_ppo', .25), ('joint_delay_acac', .20)):
                directory = root / name
                directory.mkdir()
                (directory / 'metadata.json').write_text(json.dumps(dict(status='completed',
                    phase='done', episodes_completed=2, slots_completed=8, elapsed_s=10)))
                (directory / 'test_results.json').write_text(json.dumps(dict(episodes=[
                    dict(epoch=2500000, mean_delay_s=delay, dag_count=100, failed_dags=0)])))
                runs[name] = dict(output=str(directory), pid=os.getpid())
            manifest = root / 'launch.json'
            manifest.write_text(json.dumps(dict(runs=runs, task_name='unused')))
            result = inspect(manifest, disable_task=False)
            self.assertEqual(result['status'], 'completed')
            self.assertAlmostEqual(result['comparison']['relative_delay_difference'], -.2)
            self.assertTrue((root / 'completion.json').exists())
            self.assertTrue((root / 'monitor_history.jsonl').exists())

    def test_dead_process_is_reported_without_overwriting_training_metadata(self):
        from scripts.monitor_joint_delay import inspect
        with tempfile.TemporaryDirectory(prefix='joint_monitor_') as temporary:
            root = Path(temporary)
            directory = root / 'run'
            directory.mkdir()
            original = json.dumps(dict(status='running', phase='sampling', slots_completed=8))
            (directory / 'metadata.json').write_text(original)
            manifest = root / 'launch.json'
            manifest.write_text(json.dumps(dict(runs=dict(joint_delay_ppo=dict(output=str(directory), pid=1)),
                                                task_name='unused')))
            with patch('scripts.monitor_joint_delay.process_alive', return_value=False):
                result = inspect(manifest, disable_task=False)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('attention', result['runs']['joint_delay_ppo'])
            self.assertEqual((directory / 'metadata.json').read_text(), original)


if __name__ == '__main__':
    unittest.main()
