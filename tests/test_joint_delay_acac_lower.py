import pickle
import unittest
from dataclasses import replace

import numpy as np
import torch


CONFIG = dict(hidden=16, time_dim=4, attention_heads=4, learning_rate=3e-4,
              ppo_epochs=1, minibatch_size=8, clip_ratio=.2, entropy_coef=.01,
              value_coef=.5, max_grad_norm=.5, member_gamma=1., gae_lambda=.95)


def environment():
    from env.env import MRDTSEnv
    from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
    settings = dict(channel=ChannelConfig.default(), dag=replace(DAGConfig.default(), n=3),
                    env=EnvConfig.default(), region=RegionConfig.default(),
                    uav=UAVConfig.default(), user=replace(UserConfig.default(), total_users=4))
    env = MRDTSEnv(settings)
    env.generate_scene()
    env.reset(epoch_index=0)
    return env


class TestJointDelayLower(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(45)

    def test_elapsed_gae(self):
        from methods.joint_delay_acac.learning import advantages
        adv, targets = advantages(np.array([[2., 1.], [4., 2.]]),
                                  np.array([[1., 2.], [3., 4.]]), [0., 2., 5.], .5, .8)
        np.testing.assert_allclose(adv, [[1.95, -.4], [1., -2.]], atol=1e-6)
        np.testing.assert_allclose(targets, adv + [[1., 2.], [3., 4.]])

    def test_local_histories_and_gradient(self):
        from methods.joint_delay_acac.learning import make_model
        model = make_model(3, 6, CONFIG, 'cpu')
        earlier = torch.ones(6, requires_grad=True)
        first, _, state = model.advance({0: earlier}, 0., model.initial_state())
        other, _, _ = model.advance({0: earlier, 2: earlier * 10}, 0., model.initial_state())
        torch.testing.assert_close(first[0], other[0])
        logits, values, later = model.advance({2: torch.ones(6)}, 1., state)
        torch.testing.assert_close(state[0][0], later[0][0])
        torch.testing.assert_close(state[1][0], later[1][0])
        values.sum().backward()
        self.assertGreater(float(earlier.grad.norm()), 0.)

    def test_sinusoidal_times_and_local_timestamp_effect(self):
        from methods.joint_delay_acac.learning import make_model
        model = make_model(3, 6, CONFIG, 'cpu')
        encoding = model.actor_encoders[0].encode_time(torch.tensor([0., 1.]))
        torch.testing.assert_close(encoding[0], torch.tensor([0., 1., 0., 1.]))
        obs = torch.ones(6)
        early, _, _ = model.advance({0: obs}, 0., model.initial_state())
        late, _, _ = model.advance({0: obs}, 1., model.initial_state())
        self.assertFalse(torch.equal(early[0], late[0]))

    def test_reward_discount_and_same_timestamp_collection_bucket(self):
        from methods.joint_delay_acac.rollout import bucket_rewards
        rewards, totals = bucket_rewards([0., 0., 1.], 2,
            [(0, 0., 3., 0), (1, 2., 4., 2)], .5)
        np.testing.assert_allclose(rewards, [[3., 0.], [0., 0.], [0., 2.]])
        np.testing.assert_allclose(totals, [3., 4.])

    def test_sequence_replay_matches_full_event_gradients(self):
        from methods.joint_delay_acac.learning import make_model, replay
        from methods.joint_delay_acac.rollout import collect_slot
        env = environment()
        count = env.members.member_uav_count
        model = make_model(count, 5 * count + 8, CONFIG, 'cpu')
        slot, _, _ = collect_slot(env, model, np.zeros((count, 2)), CONFIG)
        scores, _, values, _ = replay(model, slot)
        (scores.square().sum() + values.square().sum()).backward()
        expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        state = model.initial_state()
        sequential_scores, sequential_values = [], []
        self.assertTrue(any(len(e.owners) < count for e in slot.events))
        for event in slot.events:
            logits, value, state = model.advance({o: torch.as_tensor(obs) for o, obs in
                zip(event.owners, event.observations)}, event.time, state)
            sequential_values.append(value)
            for owner, mask, action in zip(event.owners, event.masks, event.actions):
                distribution = torch.distributions.Categorical(logits=logits[owner].masked_fill(~torch.as_tensor(mask), -torch.inf))
                sequential_scores.append(distribution.log_prob(torch.as_tensor(action)))
        (torch.stack(sequential_scores).square().sum() + torch.stack(sequential_values).square().sum()).backward()
        for name, p in model.named_parameters():
            if name in expected:
                torch.testing.assert_close(p.grad, expected[name], atol=2e-5, rtol=2e-4, msg=name)

    def test_local_inputs_drop_energy_queue_and_mask_other_regions(self):
        from methods.joint_delay_acac.rollout import member_inputs, order_requests
        env = environment()
        count = env.members.member_uav_count
        begin = env.begin_step(np.zeros((count, 2)))
        keys = [order[0] for order in order_requests(env._runtime, begin.requests).values()]
        observations, masks = member_inputs(env, keys)
        for row, key in enumerate(keys):
            full = env.observe_member(key)
            np.testing.assert_array_equal(observations[row], np.concatenate((full[:4 + 4 * count], full[4 + 5 * count:-1])).astype(np.float32))
            np.testing.assert_array_equal(masks[row, :count],
                env.members.region_ids[:count] == env.members.region_ids[key.owner_member_id])
            self.assertTrue(masks[row, -2:].all())

    def test_collection_replay_update(self):
        from methods.joint_delay_acac.learning import make_model, make_optimizers, replay, update
        from methods.joint_delay_acac.rollout import collect_slot
        env = environment()
        count = env.members.member_uav_count
        model = make_model(count, 5 * count + 8, CONFIG, 'cpu')
        trajectory, row, regional = collect_slot(env, model, np.zeros((count, 2)), CONFIG)
        trajectory = pickle.loads(pickle.dumps(trajectory))
        logp, _, values, _ = replay(model, trajectory)
        np.testing.assert_allclose(logp.detach(), np.concatenate([e.log_probs for e in trajectory.events]), atol=2e-6)
        np.testing.assert_allclose(values.detach(), trajectory.values, atol=2e-6)
        self.assertAlmostEqual(float(regional.sum()), row['delay_sum_s'])
        self.assertAlmostEqual(float(trajectory.rewards.sum()), row['member_reward_sum'], places=5)
        stats = update(model, make_optimizers(model, CONFIG), [trajectory], CONFIG)
        self.assertTrue(np.isfinite(stats['value_loss']))
        self.assertEqual(stats['samples'], trajectory.samples)
        if torch.cuda.is_available():
            model = model.cuda()
            stats = update(model, make_optimizers(model, CONFIG), [trajectory], CONFIG)
            self.assertTrue(np.isfinite(stats['policy_loss']))

    def test_zero_decisions_and_slot_time_origin(self):
        from methods.joint_delay_acac.learning import make_model, make_optimizers, update
        from methods.joint_delay_acac.rollout import collect_slot
        env = environment()
        count = env.members.member_uav_count
        model = make_model(count, 5 * count + 8, CONFIG, 'cpu')
        first = env.members.positions[0, :2].copy()
        other = next(position[:2].copy() for position, region in zip(env.members.positions, env.members.region_ids)
                     if region != env.members.region_ids[0])
        env.users.positions[:, :2] = other
        env.users.region_ids[:] = env.region.get_region_id(*other)
        env.members.positions[:count, :2] = first
        env._refresh_topology()
        trajectory, row, regional = collect_slot(env, model, np.zeros((count, 2)), CONFIG)
        self.assertEqual(trajectory.samples, 0)
        self.assertEqual(row['member_reward_sum'], 0.)
        self.assertEqual(trajectory.values.shape, (0, count))
        self.assertEqual(update(model, make_optimizers(model, CONFIG), [trajectory], CONFIG)['samples'], 0)
        env_a, env_b = environment(), environment()
        env_b._next_slot_start = 137.
        a, _, _ = collect_slot(env_a, model, np.zeros((count, 2)), CONFIG, deterministic=True)
        b, _, _ = collect_slot(env_b, model, np.zeros((count, 2)), CONFIG, deterministic=True)
        np.testing.assert_allclose(a.times, b.times, atol=1e-10)
        np.testing.assert_allclose(a.values, b.values, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
