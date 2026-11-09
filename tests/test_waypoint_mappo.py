import json
from pathlib import Path
import tempfile
import tomllib
import unittest

import numpy as np
import torch

from methods.hmappo.ppo import Policy
from methods.waypoint_mappo.evaluate import evaluate
from methods.waypoint_mappo.run import ROOT, deployment, train


class TestWaypointDeployment(unittest.TestCase):
    def test_only_owner_assigns_waypoint_and_zero_residual_hovers(self):
        # Each Master sees only its own UAV; the second UAV belongs to Master 0.
        observations = np.array([[0., 0., .4, .8], [.2, .7, 0., 0.]], dtype=np.float32)
        masks = np.array([[False, False, True, True], [True, True, False, False]])
        states = np.zeros((2, 1), dtype=np.float32)
        policy = Policy(4, 1, 4, True, 8)
        with torch.no_grad():
            policy.actor[-1].weight.zero_()
            policy.actor[-1].bias.zero_()
        _, hover = deployment(policy, (observations, states, masks), True)
        np.testing.assert_allclose(hover, [[.2, .7], [.4, .8]])
        with torch.no_grad():
            policy.actor[-1].bias.fill_(.7)
        _, targets = deployment(policy, (observations, states, masks), True)
        self.assertTrue(np.all((targets > hover) & (targets <= 1.)))

    def test_checkpoint_replays_paired_evaluation(self):
        with (ROOT / 'config/methods/waypoint_mappo.toml').open('rb') as stream:
            config = tomllib.load(stream)['train']
        with tempfile.TemporaryDirectory() as directory:
            source = torch.load(ROOT / config['member_checkpoint'], weights_only=True)
            source['environment']['env']['steps_per_epoch'] = 2
            checkpoint = Path(directory) / 'source.pt'
            torch.save(source, checkpoint)
            config.update(output_root=directory, member_checkpoint=str(checkpoint),
                          workers=1, updates=1, deployments=3, train_slots=2, test_epochs=[9000000])
            run = train(config)
            checkpoint.unlink()
            expected = json.loads((run / 'test_results.json').read_text())
            actual = json.loads(evaluate(run, 9000000, 1, 1).read_text())
            self.assertEqual(expected, actual)
            self.assertEqual([r['policy'] for r in actual], ['initial', 'learned', 'hover', 'random'])
            self.assertEqual(actual[1]['flight_violations'], 0)
            self.assertEqual(actual[1]['slots_completed'], 2)
