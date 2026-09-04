"""Default-size train/evaluate/resume commands tested as a user runs them."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Install requirements-rl.txt for RL tests')
class TestRLCLI(unittest.TestCase):
    def test_train_evaluate_resume_default_scene(self):
        train = ROOT / 'scripts' / 'train_rl.py'
        evaluate = ROOT / 'scripts' / 'evaluate_rl.py'
        self.assertTrue(train.is_file() and evaluate.is_file(), 'Both RL entry points must exist')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            command = [sys.executable, str(train), '--output', str(output), '--member-updates', '1',
                       '--master-updates', '1', '--rollout-slots', '2', '--eval-slots', '1',
                       '--test-slots', '1', '--eval-every', '1', '--hidden', '32', '--epochs', '1']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output / 'latest.pt').is_file())
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['completed']['master_updates'], 1)
            self.assertEqual(report['environment']['user']['total_users'], 100)
            command = [sys.executable, str(evaluate), '--checkpoint', str(output / 'best.pt')]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('mean_delay_s', result.stdout)
            command = [sys.executable, str(train), '--resume', str(output / 'latest.pt'),
                       '--member-updates', '1', '--master-updates', '2']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['completed']['master_updates'], 2)


if __name__ == '__main__':
    unittest.main()
