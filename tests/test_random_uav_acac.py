import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

import numpy as np
import torch


class TestRandomUAVACAC(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(45)

    def test_macro_gae_uses_elapsed_gamma_and_one_lambda_per_joint_event(self):
        from methods.random_uav_acac.learning import advantages

        values = np.array([[1., 2.], [3., 4.]])
        adv, targets = advantages([[2., 1.], [4., 2.]], values,
                                  [0., 2., 5.], .5, .8)
        np.testing.assert_allclose(adv, [[1.95, -.4], [1., -2.]], atol=1e-6)
        np.testing.assert_allclose(targets, adv + values)
        _, returns = advantages([[2., 1.], [4., 2.]], values,
                                [0., 2., 5.], .5, 1.)
        np.testing.assert_allclose(returns, [[3., 1.5], [4., 2.]])

    def test_only_new_observations_update_history_and_actor_is_local(self):
        from methods.random_uav_acac.network import ACAC

        model = ACAC(6, 3, hidden=16)
        state = model.initial_state()
        inputs = {0: torch.ones(6), 2: torch.zeros(6)}
        first, values, state = model.advance(inputs, 0., state)
        _, _, changed = model.advance({2: torch.ones(6)}, 2., state)
        self.assertTrue(torch.equal(state[0][0], changed[0][0]))
        self.assertTrue(torch.equal(state[1][0], changed[1][0]))
        self.assertTrue(torch.equal(state[1][1], changed[1][1]))
        self.assertFalse(torch.equal(state[1][2], changed[1][2]))
        other, _, _ = model.advance({0: inputs[0], 2: torch.full((6,), 10.)},
                                    0., model.initial_state())
        torch.testing.assert_close(first[0], other[0])
        later, _, _ = model.advance(inputs, 3., model.initial_state())
        self.assertFalse(torch.equal(first[0], later[0]))
        self.assertEqual(values.shape, (3,))

    def test_replay_propagates_gradients_through_earlier_observations(self):
        from methods.random_uav_acac.network import ACAC

        model = ACAC(6, 3, hidden=16)
        earlier = torch.randn(6, requires_grad=True)
        _, _, state = model.advance({0: earlier}, 0., model.initial_state())
        logits, values, _ = model.advance({0: torch.zeros(6)}, 1., state)
        (logits[0].sum() + values.sum()).backward()
        self.assertGreater(earlier.grad.norm().item(), 0.)
        self.assertGreater(model.critic_encoders[0].gru.weight_hh_l0.grad.norm().item(), 0.)

    def test_sequence_replay_has_the_same_gradients_as_event_replay(self):
        from methods.random_uav_acac.scene import build_scene, resolved_settings
        from methods.random_uav_acac.rollout import Adapter, collect_slot
        from methods.random_uav_acac.network import ACAC
        from methods.random_uav_acac.learning import replay

        env = build_scene(resolved_settings(users=4, dag_nodes=3), episode=0)
        adapter = Adapter(env)
        model = ACAC(adapter.obs_dim, adapter.count, hidden=16)
        trajectory, _ = collect_slot(adapter, model)
        logps, _, values, _ = replay(model, trajectory)
        (logps.square().sum() + values.square().sum()).backward()
        expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        state = model.initial_state()
        scores, estimates = [], []
        for event in trajectory.events:
            logits, value, state = model.advance(
                {o: torch.as_tensor(obs) for o, obs in zip(event.owners, event.observations)}, event.time, state)
            estimates.append(value)
            for owner, mask, action in zip(event.owners, event.masks, event.actions):
                distribution = torch.distributions.Categorical(logits=logits[owner].masked_fill(
                    ~torch.as_tensor(mask), -torch.inf))
                scores.append(distribution.log_prob(torch.as_tensor(action)))
        (torch.stack(scores).square().sum() + torch.stack(estimates).square().sum()).backward()
        for name, p in model.named_parameters():
            if name in expected:
                torch.testing.assert_close(p.grad, expected[name], atol=2e-5, rtol=2e-4, msg=name)

    def test_fixed_actions_produce_identical_environment_outcomes_to_reference(self):
        from methods.random_uav_acac.scene import build_scene, resolved_settings
        from methods.random_uav_acac.rollout import Adapter, collect_slot
        from methods.random_uav_ppo.rollout import Adapter as Reference, collect_slot as reference_collect
        from methods.random_uav_ppo.flight import derive_flight_seed

        settings = resolved_settings(users=4, dag_nodes=3)
        env, other = (build_scene(settings, episode=17) for _ in range(2))
        adapter = Adapter(env)
        reference = Reference(other, flight_seed=derive_flight_seed(45))
        for slot in range(3):
            trajectory, row = collect_slot(adapter, None, selector=lambda key, obs, mask: adapter.count)
            samples, expected = reference_collect(reference, None,
                member_selector=lambda key, mask: adapter.count)
            for key in ('mean_delay_s', 'total_energy_j', 'global_delay_reward', 'member_reward_sum', 'member_decisions'):
                self.assertAlmostEqual(row[key], expected[key], places=10, msg=key)
            for owner, decisions in samples.items():
                self.assertAlmostEqual(row[f'member_{owner}_reward_sum'], sum(s.reward for s in decisions))

    def test_flight_ers_and_local_inputs_match_reference(self):
        from methods.random_uav_acac.scene import build_scene, resolved_settings
        from methods.random_uav_acac.rollout import Adapter
        from methods.random_uav_acac.scheduling import order_requests
        from methods.random_uav_ppo.rollout import Adapter as Reference
        from methods.random_uav_ppo.ers import ERS
        from methods.random_uav_ppo.flight import derive_flight_seed

        settings = resolved_settings(users=4, dag_nodes=3)
        env, reference_env = (build_scene(settings, episode=8) for _ in range(2))
        adapter = Adapter(env, seed=45)
        reference = Reference(reference_env, flight_seed=derive_flight_seed(45))
        np.testing.assert_array_equal(env.users.positions, reference_env.users.positions)
        for slot in range(25):
            a = adapter.flight.actions(env, slot).copy()
            b = reference.flight.actions(reference_env, slot).copy()
            np.testing.assert_array_equal(a, b)
            env.members.apply_flight_actions(a, env.region.config.side_length)
            reference_env.members.apply_flight_actions(b, env.region.config.side_length)
        env.reset(epoch_index=8)
        begin = env.begin_step(adapter.flight.actions(env, 0))
        expected = ERS(env._runtime).plan(begin.requests).member_orders
        actual = order_requests(env._runtime, begin.requests)
        self.assertEqual(actual, {node.index: keys for node, keys in expected.items()})
        keys = [keys[0] for keys in actual.values()]
        observations, masks = adapter.inputs(keys)
        ref_obs, _, ref_masks = Reference(env).member_inputs(keys)
        np.testing.assert_array_equal(observations, ref_obs)
        np.testing.assert_array_equal(masks, ref_masks)

    def test_collect_rewards_and_replay_match_behavior_policy(self):
        from methods.random_uav_acac.scene import build_scene, resolved_settings
        from methods.random_uav_acac.rollout import Adapter, collect_slot
        from methods.random_uav_acac.network import ACAC
        from methods.random_uav_acac.learning import replay, update

        env = build_scene(resolved_settings(users=4, dag_nodes=3), episode=0)
        adapter = Adapter(env)
        model = ACAC(adapter.obs_dim, adapter.count, hidden=16)
        trajectory, metrics = collect_slot(adapter, model)
        self.assertEqual(sum(len(e.owners) for e in trajectory.events), metrics['member_decisions'])
        self.assertAlmostEqual(float(trajectory.rewards.sum()), metrics['member_reward_sum'], places=6)
        logp, _, values, _ = replay(model, trajectory)
        expected = np.concatenate([e.log_probs for e in trajectory.events])
        np.testing.assert_allclose(logp.detach().numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(values.detach().numpy(), trajectory.values, atol=1e-6)
        self.assertTrue(any(len(e.owners) < adapter.count for e in trajectory.events))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
        stats = update(model, optimizer, [trajectory], ppo_epochs=2, minibatch_size=8)
        self.assertTrue(np.isfinite(stats['value_loss']))
        for prefix in ('actors.', 'actor_encoders.', 'critic_encoders.', 'attention.', 'value_head.'):
            self.assertTrue(any(not torch.equal(v, model.state_dict()[k])
                                for k, v in before.items() if k.startswith(prefix)), prefix)

    def test_no_member_decisions_is_a_valid_empty_rollout(self):
        from methods.random_uav_acac.scene import build_scene, resolved_settings
        from methods.random_uav_acac.rollout import Adapter, collect_slot
        from methods.random_uav_acac.network import ACAC
        from methods.random_uav_acac.learning import update

        env = build_scene(resolved_settings(users=4, dag_nodes=3), episode=0)
        # Separate users and UAVs into two existing regions, so all DAGs execute locally.
        first_region = env.members.region_ids[0]
        other = next(p.copy() for p, r in zip(env.members.positions, env.members.region_ids)
                     if r != first_region)
        env.users.positions[:, :2] = other[:2]
        env.users.region_ids[:] = env.region.get_region_id(*other[:2])
        env.members.positions[:env.members.member_uav_count, :2] = env.members.positions[0, :2]
        env._refresh_topology()
        adapter = Adapter(env)
        model = ACAC(adapter.obs_dim, adapter.count, hidden=16)
        trajectory, metrics = collect_slot(adapter, model)
        self.assertEqual(metrics['member_decisions'], 0)
        self.assertEqual(metrics['dag_count'], 4)
        optimizer = torch.optim.Adam(model.parameters())
        self.assertEqual(update(model, optimizer, [trajectory])['samples'], 0)

    def test_defaults_and_short_training_with_paired_evaluation(self):
        from methods.random_uav_acac.run import train, load_checkpoint
        from methods.random_uav_acac.evaluate import compare
        from methods.random_uav_ppo.run import train as reference_train

        configs = []
        for name in ('random_uav_acac', 'random_uav_ppo'):
            with Path(f'config/methods/{name}.toml').open('rb') as stream:
                configs.append(tomllib.load(stream)['train'])
        for key in configs[1]:
            self.assertEqual(configs[0][key], configs[1][key], key)
        with tempfile.TemporaryDirectory() as root:
            options = dict(episodes=1, slots=2, users=4, dag_nodes=3, hidden=16,
                           ppo_epochs=1, rollout_slots=2, output_root=root)
            out = train(**options)
            other = reference_train(**options)
            from methods.random_uav_acac.evaluate import PPOCheckpoint
            from methods.random_uav_ppo.run import load_checkpoint as load_ppo
            ref_env, ref_policies, ref_saved = load_ppo(other / 'checkpoint.pt')
            reader = PPOCheckpoint(ref_saved, 5 * ref_env.members.member_uav_count + 8,
                                   ref_env.members.member_uav_count, 'cpu')
            for actor, policy in zip(reader.actors, ref_policies):
                inputs = torch.randn(5 * ref_env.members.member_uav_count + 8)
                torch.testing.assert_close(actor(inputs), policy.actor(inputs))
            _, trained, checkpoint = load_checkpoint(out / 'checkpoint.pt')
            _, initial, _ = load_checkpoint(out / 'initial.pt')
            self.assertEqual(checkpoint['episodes_completed'], 1)
            self.assertTrue(any(not torch.equal(v, trained.state_dict()[k])
                                for k, v in initial.state_dict().items()))
            self.assertEqual(json.loads((out / 'metadata.json').read_text())['status'], 'completed')
            with (out / 'updates.csv').open() as stream:
                update_row = next(csv.DictReader(stream))
            self.assertEqual(sum(int(update_row[f'member_{i}_samples']) for i in range(ref_env.members.member_uav_count)), 24)
            result = compare([out / 'initial.pt', out / 'checkpoint.pt'],
                             ppo_checkpoints=[other / 'checkpoint.pt'], episodes=2, output_root=root)
            with (result / 'episodes.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)  # own, random, initial, trained, reference PPO
            for episode in (1000000, 1000001):
                paired = [r for r in rows if int(r['episode']) == episode]
                self.assertEqual(len({r['dag_stream_sha256'] for r in paired}), 1)
                self.assertEqual(len({r['flight_stream_sha256'] for r in paired}), 1)


if __name__ == '__main__':
    unittest.main()
