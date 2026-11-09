import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from methods.joint_delay_ppo.run import main, train
from methods.joint_delay_ppo.support import load_config, write_json


class TestJointDelayPPOResume(unittest.TestCase):
    def test_cli_loads_resume_configuration_before_episode_override(self):
        config = load_config()
        with patch('sys.argv', ['joint_delay_ppo', '--resume', 'saved.pt', '--episodes', '150',
                                '--output-dir', 'continued']), \
                patch('methods.joint_delay_ppo.run.torch.load', return_value={'config': config}) as load, \
                patch('methods.joint_delay_ppo.run.train') as training:
            main()
        load.assert_called_once_with(Path('saved.pt'), weights_only=True, map_location='cpu')
        self.assertEqual(training.call_args.args[0]['train']['episodes'], 150)
        self.assertEqual(training.call_args.kwargs['resume'], Path('saved.pt'))

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, wanted in zip(actual, expected):
                self.assert_state_equal(value, wanted)
        else:
            self.assertEqual(actual, expected)

    def test_resume_odd_boundary_restores_training_state_and_continues_budget(self):
        config = load_config()
        config['train'].update(episodes=1, slots=2, workers=2, rollout_slots=4, users=4,
                               dag_nodes=3, hidden=8, ppo_epochs=1, evaluation_slots=1,
                               validation_epochs=[1500000], test_epochs=[2500000],
                               validation_interval=2)
        config['high'].update(hidden=8, replay_size=16, batch_size=2, fit_interval=1,
                              fit_steps=1, rounds=1, warmup_samples=1)
        with tempfile.TemporaryDirectory() as temporary:
            original, continued = Path(temporary) / 'original', Path(temporary) / 'continued'
            train(config, original)
            source = original / 'latest.pt'
            saved = torch.load(source, weights_only=True)
            expected_best = torch.load(original / 'best.pt', weights_only=True)
            self.assertTrue(any(state['state'] for state in saved['low_optimizers']))
            self.assertTrue(all(state['optimizer']['state'] for state in saved['high']))
            self.assertTrue(all(state['samples'] == 2 for state in saved['high']))
            # A candidate saved before validation can contain stale selection fields.
            saved.update(best_validation_delay_s=-1., selected_episode=999)
            saved['cuda_rng'] = torch.get_rng_state()
            torch.save(saved, source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with (original / 'episodes.csv').open() as stream:
                previous_elapsed = list(csv.DictReader(stream))[-1]['elapsed_s']
            for name, counter in (('episodes.csv', 'episode'), ('updates.csv', 'update')):
                with (original / name).open() as stream:
                    reader = csv.DictReader(stream)
                    fields = reader.fieldnames
                    future = list(reader)[-1]
                future[counter] = '999'
                with (original / name).open('a', newline='') as stream:
                    csv.DictWriter(stream, fieldnames=fields).writerow(future)
            (original / 'validation_9999.json').write_text('[]', encoding='utf-8')
            source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in original.iterdir() if path.is_file()}
            config = copy.deepcopy(config)
            config['train']['episodes'] = 3
            metadata_attempts = []
            def busy_metadata_once(path, value):
                if path.name == 'metadata.json':
                    metadata_attempts.append(dict(value))
                    if len(metadata_attempts) == 1:
                        raise PermissionError('reader holds metadata')
                write_json(path, value)
            with patch('methods.joint_delay_ppo.run.torch.cuda.set_rng_state') as restore_cuda, \
                    patch('methods.joint_delay_ppo.run.write_json', side_effect=busy_metadata_once), \
                    self.assertWarnsRegex(RuntimeWarning, 'Metadata update postponed'):
                train(config, continued, resume=source)
            restore_cuda.assert_called_once()
            torch.testing.assert_close(restore_cuda.call_args.args[0], saved['cuda_rng'])
            self.assertEqual(metadata_attempts[0]['resumed_from'], str(source.resolve()))
            self.assertEqual(metadata_attempts[0]['selected_episode'], expected_best['selected_episode'])
            self.assertEqual(metadata_attempts[0]['best_validation_delay_s'],
                             expected_best['best_validation_delay_s'])
            resumed = torch.load(continued / 'resume_initial.pt', weights_only=True)
            for key in ('low', 'low_optimizers', 'high', 'high_observations', 'high_rng', 'torch_rng',
                        'episodes_completed', 'slots_completed', 'updates'):
                self.assert_state_equal(resumed[key], saved[key])
            self.assertEqual(resumed['selected_episode'], expected_best['selected_episode'])
            self.assertEqual(resumed['best_validation_delay_s'], expected_best['best_validation_delay_s'])
            self.assertEqual((continued / 'initial.pt').read_bytes(), (original / 'initial.pt').read_bytes())
            final = torch.load(continued / 'latest.pt', weights_only=True)
            boundary = torch.load(continued / 'boundary.pt', weights_only=True)
            self.assertEqual(final['episodes_completed'], 3)
            self.assertEqual(final['slots_completed'], 6)
            self.assertEqual(final['updates'], saved['updates'] + 1)
            self.assertEqual(final['high_observations'], 6)
            self.assertEqual(boundary['slots_completed'], boundary['episodes_completed'] * 2)
            with (continued / 'episodes.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row['episode']) for row in rows], [1, 2, 3])
            self.assertGreater(float(rows[-1]['elapsed_s']), float(previous_elapsed))
            with (continued / 'updates.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), final['updates'])
            self.assertTrue((continued / 'validation_0000.json').exists())
            self.assertTrue((continued / 'validation_0001.json').exists())
            self.assertFalse((continued / 'validation_0002.json').exists())
            self.assertFalse((continued / 'validation_9999.json').exists())
            self.assertTrue((continued / 'validation_0003.json').exists())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            self.assertEqual({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in original.iterdir() if path.is_file()}, source_hashes)
            self.assertEqual(json.loads((continued / 'metadata.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
