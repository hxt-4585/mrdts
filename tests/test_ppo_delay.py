"""Recovered PPO behavior on the current scene and decimal-Kbit runtime."""

from dataclasses import replace
import importlib
import unittest

import numpy as np
import torch

from experiments.config import load_config
from experiments.randomness import RandomStreams
from experiments.scene import build_scene, resolved_settings


def scene_for(episode=0, users=4):
    config = replace(load_config(), users=users, dag_nodes=3, slots=3)
    return build_scene(resolved_settings(config), randomness=RandomStreams.from_config(config, episode))


class TestPPODelay(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def api(self):
        try:
            return importlib.import_module('methods.solutions.ppo_delay.method'), importlib.import_module('methods.solutions.ppo_delay.rollout')
        except ModuleNotFoundError:
            self.fail('The migrated PPO method and rollout are missing')

    def test_masks_and_ground_prior(self):
        self.api()
        from methods.solutions.ppo_delay.networks import MemberActor, MasterActor
        actor = MemberActor(23)
        obs = torch.zeros(2, 14, 23)
        obs[:, 0, 11] = 1
        mask = torch.zeros(2, 14, dtype=torch.bool)
        mask[:, [0, 2, 13]] = True
        dist = actor.distribution(obs, mask)
        self.assertTrue((dist.probs[:, 0] > .8).all())
        self.assertTrue((dist.probs[~mask] == 0).all())
        self.assertTrue((dist.probs[mask] > 0).all())
        master = MasterActor(4, 6)
        owned = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        latent = torch.zeros(1, 6)
        before = master.evaluate(torch.zeros(1, 4), owned, latent)[0]
        latent[~owned] = 99
        torch.testing.assert_close(before, master.evaluate(torch.zeros(1, 4), owned, latent)[0])

    def test_kbit_observation_matches_runtime_input(self):
        method, _ = self.api()
        from methods.solutions.ppo_delay.observations import MemberPlanning
        scene = scene_for()
        env = method.PlanningEnvironment(scene)
        batch = env.begin(scene.workload(0), np.zeros((12, 2)))
        planning = MemberPlanning(env, batch)
        key = batch.plan.order[0]
        observation, mask = planning.observe(key)
        request = planning.requests[key.owner_member_id, key.user_id, key.dag_id]
        owner_action = key.owner_member_id + 1
        route = batch.runtime.route_planner.input_route(request.ground_device, request.owner_member, request.owner_member)
        expected = sum(batch.runtime.transfer_duration_s(hop, request.dag.node_features[key.node_id, 0] * 1000) for hop in route.hops)
        self.assertTrue(mask[owner_action])
        self.assertAlmostEqual(float(observation[owner_action, 17]), expected, places=6)
        outcome, metrics = env.finish(batch, {key: 0 for key in batch.plan.order})
        self.assertEqual(len(outcome.result.dags), 4)
        self.assertEqual(metrics['failure_rate'], 0)
        self.assertAlmostEqual(metrics['reward'], -metrics['mean_delay_s'])

    def test_epoch_flush_preserves_positions_and_frozen_actor(self):
        method, rollout = self.api()
        scene = scene_for()
        learner = rollout.Learner(method.PlanningEnvironment(scene), device='cpu', hidden=16)
        with torch.no_grad():
            learner.master.mean[-1].weight.zero_()
            learner.master.mean[-1].bias.fill_(.1)
        start = scene.simulator.members.positions.copy()
        frozen = [p.detach().clone() for p in learner.master.parameters()]
        member_before = [p.detach().clone() for p in learner.member.parameters()]
        rows, updates = [], []
        result = rollout.run_epoch(scene, learner, stage='member', steps=3, update_every=2,
                                   on_step=rows.append, on_update=updates.append)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(updates), 2)
        self.assertEqual([r['slot_start'] for r in rows], [0., 1., 2.])
        np.testing.assert_allclose(scene.simulator.members.positions[:12, :2], start[:12, :2] + np.tanh(.1)*15, atol=2e-4)
        self.assertAlmostEqual(result['epoch_return'], sum(r['reward'] for r in rows))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(frozen, learner.master.parameters())))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(member_before, learner.member.parameters())))
        frozen = [p.detach().clone() for p in learner.member.parameters()]
        rollout.run_epoch(scene_for(1), learner, stage='master', steps=3)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(frozen, learner.member.parameters())))

    def test_true_terminal_gae_and_all_failed_reward(self):
        self.api()
        from methods.learning.algorithms.ppo import gae_returns
        from methods.solutions.ppo_delay.reward import delay_metrics
        from env.runtime.slot_result import SlotResult, DAGResult
        np.testing.assert_allclose(gae_returns(np.array([[-1.], [-2.], [-8.]]), np.zeros((3,1)), [False,True,True], lam=1.)[:,0], [-3,-2,-8])
        metrics = delay_metrics(SlotResult(5,6,(DAGResult((0,0,0),None,True),),0,0))
        self.assertEqual(metrics['reward'], -1)


if __name__ == '__main__':
    unittest.main()
