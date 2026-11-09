import csv
import json
import io
from contextlib import redirect_stdout
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from methods.random_uav_ppo.rollout import Adapter
from methods.random_uav_ppo.support import build_scene, resolved_settings


class TestFrozenMemberPPO(unittest.TestCase):
    def test_training_freezes_members_and_checkpoint_is_self_contained(self):
        from methods.frozen_member_ppo.run import train, load_checkpoint
        from methods.frozen_member_ppo.evaluate import compare

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='frozen_member_ppo_') as root:
            source_path = Path(root) / 'member.pt'
            settings = resolved_settings(users=4, dag_nodes=3)
            settings['env'] = replace(settings['env'], steps_per_epoch=5)
            members = Adapter(build_scene(settings, episode=0)).policies(16, 'cpu')
            original = {k: v.clone() for k, v in members.state_dict().items()}
            torch.save(dict(method='random_uav_ppo', config=dict(hidden=16, flight_seed=46,
                resample_slots=10), environment={k: asdict(v) for k, v in settings.items()},
                episodes_completed=371, member_policies=original), source_path)
            console = io.StringIO()
            with redirect_stdout(console):
                output = train(member_checkpoint=source_path, episodes=2, rollout_slots=3,
                               ppo_epochs=1, minibatch_size=8, output_root=root)
            env, master, frozen, saved = load_checkpoint(output / 'checkpoint.pt')
            initial = torch.load(output / 'initial.pt', weights_only=True)
            self.assertEqual(output.parent.parent.name, 'frozen_member_ppo')
            self.assertEqual(saved['episodes_completed'], 2)
            self.assertEqual(saved['updates'], 4)
            self.assertEqual(saved['member_source']['episodes_completed'], 371)
            self.assertEqual(env.users.num_users, 4)
            self.assertFalse(frozen.training)
            self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
            self.assertNotIn('member_optimizers', saved)
            for key, value in frozen.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]), key)
            self.assertTrue(any(not torch.equal(v, initial['master'][k])
                                for k, v in master.state_dict().items() if k.startswith('actor.')))
            self.assertTrue(all(torch.isfinite(v).all() for v in master.state_dict().values()))
            self.assertFalse((output / 'updates.csv').exists())
            self.assertFalse((output / 'metrics.csv').exists())
            with (output / 'episodes.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            lines = [line for line in console.getvalue().splitlines() if line.startswith('Episode ')]
            self.assertEqual(len(lines), 2)
            for row, line in zip(rows, lines):
                expected = 4 * 5 * float(row['master_training_reward_mean'])
                self.assertAlmostEqual(float(row['episode_reward_sum']), expected)
                self.assertAlmostEqual(float(row['episode_reward_sum']),
                    sum(float(row[f'master_{i}_reward_sum']) for i in range(4)))
                self.assertIn(f'Reward sum {expected:.4f}', line)
                self.assertIn(f'Mean delay {float(row["mean_delay_s"]):.4f}s', line)
            self.assertEqual(json.loads((output / 'metadata.json').read_text())['status'], 'completed')
            source_path.unlink()
            result = compare(output / 'checkpoint.pt', episodes=2, output_root=root)
            with (result / 'episodes.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual({r['policy'] for r in rows}, {'learned', 'random', 'hover'})
            self.assertEqual({int(r['episode']) for r in rows}, {1000000, 1000001})
            for policy in ('learned', 'random', 'hover'):
                self.assertTrue(all(int(r['slots_completed']) == 5 for r in rows if r['policy'] == policy))
            _, _, reloaded, _ = load_checkpoint(output / 'checkpoint.pt')
            self.assertTrue(all(torch.equal(v, original[k]) for k, v in reloaded.state_dict().items()))
