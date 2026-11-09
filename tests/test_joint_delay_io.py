import importlib
import json
import os
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULES = ('methods.joint_delay_ppo.support', 'methods.joint_delay_acac.support',
           'scripts.monitor_joint_delay')


class TestJointDelayIO(unittest.TestCase):
    def test_transient_permission_errors_preserve_old_json_until_success(self):
        for name in MODULES:
            with self.subTest(module=name), tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module(name)
                destination = Path(directory) / 'metadata.json'
                destination.write_text('{"status": "old"}', encoding='utf-8')
                original_replace = os.replace
                attempts = []
                def replace(source, target):
                    attempts.append((source, target))
                    self.assertEqual(json.loads(destination.read_text(encoding='utf-8')), {'status': 'old'})
                    self.assertEqual(json.loads(Path(source).read_text(encoding='utf-8')), {'status': 'new'})
                    if len(attempts) < 3:
                        raise PermissionError('temporarily open by a reader')
                    original_replace(source, target)
                with patch.object(os, 'replace', side_effect=replace), patch.object(time, 'sleep') as sleep:
                    module.write_json(destination, {'status': 'new'})
                self.assertEqual(len(attempts), 3)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [.02, .04])
                self.assertEqual(json.loads(destination.read_text(encoding='utf-8')), {'status': 'new'})
                self.assertFalse(destination.with_suffix('.json.tmp').exists())

    def test_non_permission_error_is_not_retried(self):
        for name in MODULES:
            with self.subTest(module=name), tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module(name)
                source, target = Path(directory) / 'source', Path(directory) / 'target'
                source.write_text('new', encoding='utf-8')
                target.write_text('old', encoding='utf-8')
                with patch.object(os, 'replace', side_effect=FileNotFoundError('missing')) as replace, \
                        patch.object(time, 'sleep') as sleep:
                    with self.assertRaises(FileNotFoundError):
                        module.replace_file(source, target)
                self.assertEqual(replace.call_count, 1)
                sleep.assert_not_called()
                self.assertEqual(target.read_text(encoding='utf-8'), 'old')

    def test_persistent_permission_error_has_bounded_retries(self):
        for name in MODULES:
            with self.subTest(module=name), tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module(name)
                target = Path(directory) / 'metadata.json'
                target.write_text('{"status": "old"}', encoding='utf-8')
                with patch.object(os, 'replace', side_effect=PermissionError('busy')) as replace, \
                        patch.object(time, 'sleep') as sleep:
                    with self.assertRaises(PermissionError):
                        module.write_json(target, {'status': 'new'})
                self.assertEqual(replace.call_count, 8)
                self.assertEqual(sleep.call_count, 7)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [.02, .04, .08, .16, .32, .5, .5])
                self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'status': 'old'})


if __name__ == '__main__':
    unittest.main()
