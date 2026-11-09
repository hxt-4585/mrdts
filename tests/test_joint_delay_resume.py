import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import torch


class ResumeTests(unittest.TestCase):
    def test_history_discards_records_after_restored_boundary(self):
        from methods.joint_delay_acac.run import copy_completed_history
        with tempfile.TemporaryDirectory(prefix='joint_history_') as temporary:
            source, destination = Path(temporary) / 'old', Path(temporary) / 'new'
            source.mkdir()
            destination.mkdir()
            (source / 'episodes.csv').write_text('episode,elapsed_s\n1,10\n2,20\n3,30\n', encoding='utf-8')
            (source / 'updates.csv').write_text('update,slots_completed\n1,4\n2,8\n3,12\n', encoding='utf-8')
            for name in ('best.pt', 'initial.pt', 'validation_0000.json', 'validation_0002.json', 'validation_0003.json'):
                (source / name).write_bytes(b'fixture')
            elapsed = copy_completed_history(source, destination, 2, 2)
            self.assertEqual(elapsed, 20)
            self.assertEqual((destination / 'episodes.csv').read_text().splitlines(),
                             ['episode,elapsed_s', '1,10', '2,20'])
            self.assertEqual((destination / 'updates.csv').read_text().splitlines(),
                             ['update,slots_completed', '1,4', '2,8'])
            self.assertFalse((destination / 'validation_0003.json').exists())

    def test_acac_resumes_weights_optimizer_replay_budget_and_history(self):
        from methods.joint_delay_acac import run
        from methods.joint_delay_acac.support import load_config
        config = load_config()
        config['train'].update(episodes=2, slots=2, workers=2, rollout_slots=4, hidden=16,
            ppo_epochs=1, device='cuda' if torch.cuda.is_available() else 'cpu', users=4, dag_nodes=3, validation_interval=2,
            validation_epochs=[1500000], test_epochs=[2500000], evaluation_slots=1)
        config['high'].update(hidden=16, replay_size=16, fit_interval=1, warmup_samples=1)
        with tempfile.TemporaryDirectory(prefix='joint_resume_smoke_') as temporary:
            source = run.train(config, Path(temporary) / 'original')
            checkpoint = source / 'candidate.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            best = torch.load(source / 'best.pt', map_location='cpu', weights_only=True)
            # A pre-validation candidate can carry an older selection record.
            saved['best_validation_delay_s'] = best['best_validation_delay_s'] + .1
            saved['selected_episode'] = 0
            torch.save(saved, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            extended = copy.deepcopy(config)
            extended['train']['episodes'] = 4
            observed = []
            real_update = run.update
            real_write_json = run.write_json
            lock_failures = []

            def temporarily_locked_status(path, value):
                if Path(path).name == 'metadata.json' and not lock_failures:
                    lock_failures.append(True)
                    raise PermissionError('simulated exhausted transient lock retries')
                return real_write_json(path, value)

            def check_restoration(model, optimizer, trajectories, options):
                if not observed:
                    for name, tensor in model.state_dict().items():
                        torch.testing.assert_close(tensor.cpu(), saved['low'][name], rtol=0, atol=0)
                    self.assertEqual(len(optimizer.state), len(saved['low_optimizers'][0]['state']))
                    actual_steps = [int(state['step']) for state in optimizer.state.values()]
                    expected_steps = [int(state['step']) for state in saved['low_optimizers'][0]['state'].values()]
                    self.assertEqual(actual_steps, expected_steps)
                observed.append(True)
                return real_update(model, optimizer, trajectories, options)

            with patch.object(run, 'update', side_effect=check_restoration), \
                 patch.object(run, 'write_json', side_effect=temporarily_locked_status), \
                 warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                output = run.train(extended, Path(temporary) / 'continued', resume=checkpoint)
            self.assertTrue(any('training continues' in str(item.message) for item in captured))
            resumed = torch.load(output / 'resume_initial.pt', map_location='cpu', weights_only=True)
            self.assertEqual(resumed['best_validation_delay_s'], best['best_validation_delay_s'])
            self.assertEqual(resumed['selected_episode'], best['selected_episode'])
            final = torch.load(output / 'latest.pt', map_location='cpu', weights_only=True)
            self.assertEqual(final['episodes_completed'], 4)
            self.assertEqual(final['slots_completed'], 8)
            self.assertEqual(final['updates'], saved['updates'] + 1)
            self.assertEqual(final['high_observations'], 8)
            boundary = torch.load(output / 'boundary.pt', map_location='cpu', weights_only=True)
            self.assertEqual(boundary['slots_completed'], boundary['episodes_completed'] * config['train']['slots'])
            self.assertTrue(all(state['samples'] == 8 for state in final['high']))
            with (output / 'episodes.csv').open(newline='', encoding='utf-8') as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual([int(row['episode']) for row in history], [1, 2, 3, 4])
            self.assertTrue((output / 'validation_0000.json').exists())
            self.assertTrue((output / 'initial.pt').exists())
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)
            metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(metadata['resumed_from'], str(checkpoint.resolve()))


if __name__ == '__main__':
    unittest.main()
