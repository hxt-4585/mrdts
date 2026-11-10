from dataclasses import replace
import pickle
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from env.env import MRDTSEnv
from env.contracts import DAGRequest
from env.types import DirectedChannelKey, EntityKind, EntityRef, TaskKey
from env.workload.dag_generator import DAG
from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig


CONFIG = dict(hidden=16, learning_rate=0.003, ppo_epochs=2, minibatch_size=8,
              clip_ratio=.2, entropy_coef=.01, value_coef=.5, max_grad_norm=.5,
              member_gamma=1., gae_lambda=.95)


def scene():
    settings = dict(region=RegionConfig.default(), user=replace(UserConfig.default(), total_users=4),
                    uav=UAVConfig.default(), channel=ChannelConfig.default(),
                    dag=replace(DAGConfig.default(), n=3), env=EnvConfig.default())
    env = MRDTSEnv(settings)
    env.generate_scene()
    env.reset(epoch_index=0)
    return env


class TestJointDelayPPOLower(unittest.TestCase):
    def test_policy_initialization_and_optimizer_match_independent_ppo(self):
        from methods.joint_delay_ppo.learning import make_model, make_optimizers
        model = make_model(2, 18, CONFIG, 'cpu')
        for policy in model:
            for network, final_gain in ((policy.actor, .01), (policy.critic, 1.)):
                layers = [layer for layer in network if isinstance(layer, torch.nn.Linear)]
                for index, layer in enumerate(layers):
                    gain = final_gain if index == len(layers) - 1 else np.sqrt(2.)
                    weight = layer.weight.detach()
                    gram = weight @ weight.T if weight.shape[0] <= weight.shape[1] else weight.T @ weight
                    torch.testing.assert_close(gram, torch.eye(gram.shape[0]) * gain**2,
                                               rtol=1e-4, atol=1e-6)
                    torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))
        self.assertTrue(all(optimizer.defaults['eps'] == 1e-5
                            for optimizer in make_optimizers(model, CONFIG)))

    def test_ers_weights_each_core_and_sums_all_route_hops(self):
        from methods.joint_delay_ppo.rollout import rank_orders
        owner = EntityRef(EntityKind.MEMBER_UAV, 0)
        ground = EntityRef(EntityKind.GROUND_DEVICE, 0)
        bs = EntityRef(EntityKind.BS, 0)
        relay = EntityRef(EntityKind.MEMBER_UAV, 1)
        nodes = (owner, bs, ground)
        long_route = False
        def route(source, destination, user, member):
            if source == destination:
                hops = ()
            elif long_route:
                hops = (DirectedChannelKey(source, relay), DirectedChannelKey(relay, destination))
            else:
                hops = (DirectedChannelKey(source, destination),)
            return SimpleNamespace(hops=hops)
        runtime = SimpleNamespace(
            candidate_execution_nodes=lambda member, user: nodes,
            servers={node: SimpleNamespace(core_frequencies=frequencies)
                     for node, frequencies in zip(nodes, ((1., 1.), (10.,), (1.,)))},
            route_planner=SimpleNamespace(predecessor_route=route),
            transfer_duration_s=lambda hop, bits: bits)
        single = DAG(1, [], np.array([[0., 2.]]), {})
        chain = DAG(2, [(0, 1)], np.array([[0., .2], [0., .2]]), {(0, 1): .0019})
        requests = [DAGRequest(1, chain, owner, ground), DAGRequest(0, single, owner, ground)]
        # Four cores give .775 s/cycle and .625 s/bit (same-device pairs cost zero).
        self.assertEqual(rank_orders(runtime, requests)[0],
                         [TaskKey(0, 0, 0, 0), TaskKey(0, 0, 1, 0), TaskKey(0, 0, 1, 1)])
        long_route = True
        self.assertEqual(rank_orders(runtime, requests)[0],
                         [TaskKey(0, 0, 1, 0), TaskKey(0, 0, 0, 0), TaskKey(0, 0, 1, 1)])

    def test_local_models_and_zero_decisions_leave_parameters_untouched(self):
        from methods.joint_delay_ppo.learning import make_model, make_optimizers, update
        model = make_model(2, 18, CONFIG, 'cpu')
        self.assertFalse(set(map(id, model[0].parameters())) & set(map(id, model[1].parameters())))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        stats = update(model, make_optimizers(model, CONFIG), [{0: [], 1: []}], CONFIG)
        self.assertEqual(stats['samples'], 0)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))

    def test_gae_uses_terminal_zero_and_individual_decision_sequence(self):
        from methods.joint_delay_ppo.learning import Decision, finish_trajectory
        samples = [Decision(np.zeros(3), np.ones(2, bool), 0, 0., value, reward)
                   for value, reward in [(2., 1.), (3., 4.)]]
        finish_trajectory(samples, gamma=1., gae_lambda=1.)
        np.testing.assert_allclose([s.advantage for s in samples], [3., 1.])
        np.testing.assert_allclose([s.target for s in samples], [5., 4.])

    def test_real_slot_masks_energy_exclusion_rewards_and_update(self):
        from methods.joint_delay_ppo.learning import make_model, make_optimizers, update
        from methods.joint_delay_ppo.rollout import collect_slot
        env = scene()
        count = env.members.member_uav_count
        model = make_model(count, 5 * count + 8, CONFIG, 'cpu')
        before = {k: v.clone() for k, v in model.state_dict().items()}
        raw_observations = {}
        original = env.observe_member
        def record(key):
            raw_observations[key] = original(key).copy()
            return raw_observations[key]
        env.observe_member = record
        trajectory, row, delays = collect_slot(env, model, np.zeros((count, 2), np.float32), CONFIG)
        self.assertEqual(row['dag_count'], 4)
        self.assertEqual(row['failed_dags'], 0)
        self.assertEqual(row['member_decisions'], 12)
        self.assertAlmostEqual(delays.sum(), row['delay_sum_s'])
        self.assertTrue(np.isfinite(row['member_reward_sum']))
        for owner, samples in trajectory.items():
            for sample in samples:
                raw = raw_observations[sample.key]
                np.testing.assert_array_equal(sample.observation,
                    np.concatenate((raw[:4 + 4 * count], raw[4 + 5 * count:-1])).astype(np.float32))
                np.testing.assert_array_equal(sample.mask[:count],
                    env.members.region_ids[:count] == env.members.region_ids[owner])
                self.assertTrue(sample.mask[count:].all())
                self.assertEqual(sample.observation.shape, (5 * count + 8,))
        restored = pickle.loads(pickle.dumps(trajectory))
        stats = update(model, make_optimizers(model, CONFIG), [restored], CONFIG)
        self.assertEqual(stats['samples'], 12)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertTrue(any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()))

    def test_ownerless_dags_still_contribute_to_user_region_delay(self):
        from methods.joint_delay_ppo.learning import make_model
        from methods.joint_delay_ppo.rollout import collect_slot
        env = scene()
        count = env.members.member_uav_count
        # Every member serves region 1; remaining users run locally in the environment.
        env.members.positions[:count, :2] = env.members.positions[0, :2]
        model = make_model(count, 5 * count + 8, CONFIG, 'cpu')
        trajectory, row, delays = collect_slot(env, model, np.zeros((count, 2), np.float32),
                                              CONFIG, deterministic=True)
        self.assertLess(row['member_decisions'], 12)
        self.assertEqual(row['dag_count'], 4)
        self.assertTrue(np.all(delays > 0.))
        self.assertAlmostEqual(delays.sum(), row['delay_sum_s'])
        inactive = [owner for owner, samples in trajectory.items() if not samples]
        self.assertTrue(inactive)

    def test_all_ownerless_slot_has_no_decisions_and_nonzero_delay(self):
        from methods.joint_delay_ppo.learning import make_model, make_optimizers, update
        from methods.joint_delay_ppo.rollout import collect_slot
        env = scene()
        count = env.members.member_uav_count
        env.members.positions[:count, :2] = env.members.positions[0, :2]
        member_region = env.region.get_region_id(*env.members.positions[0, :2])
        user_region = member_region % env.region.config.region_count + 1
        env.users.region_ids[:] = user_region
        model = make_model(count, 5 * count + 8, CONFIG, 'cpu')
        trajectory, row, delays = collect_slot(env, model, np.zeros((count, 2), np.float32), CONFIG)
        self.assertEqual(row['member_decisions'], 0)
        self.assertEqual(row['dag_count'], 4)
        self.assertGreater(delays[user_region - 1], 0.)
        self.assertEqual(np.count_nonzero(delays), 1)
        self.assertEqual(update(model, make_optimizers(model, CONFIG), [trajectory], CONFIG)['samples'], 0)

    def test_delayed_rewards_stay_attached_to_original_task_keys(self):
        from methods.joint_delay_ppo.learning import make_model
        from methods.joint_delay_ppo.rollout import collect_slot
        env = scene()
        count = env.members.member_uav_count
        model = make_model(count, 5 * count + 8, CONFIG, 'cpu')
        received = {}
        pop, end = env.pop_member_rewards, env.end_step
        def record_pop():
            rewards = pop()
            received.update(rewards)
            return rewards
        def record_end():
            result = end()
            received.update(result.member_rewards)
            return result
        env.pop_member_rewards, env.end_step = record_pop, record_end
        gradient_flags = []
        handles = [policy.actor.register_forward_pre_hook(
            lambda module, args: gradient_flags.append(torch.is_grad_enabled())) for policy in model]
        trajectory, row, _ = collect_slot(env, model, np.zeros((count, 2), np.float32), CONFIG)
        for handle in handles:
            handle.remove()
        samples = [sample for sequence in trajectory.values() for sample in sequence]
        self.assertEqual({sample.key: sample.reward for sample in samples},
                         {key: reward.compute_delay() for key, reward in received.items()})
        self.assertEqual(len(received), row['member_decisions'])
        self.assertFalse(any(gradient_flags))
        active = {owner for owner, sequence in trajectory.items() if sequence}
        before = [{key: value.clone() for key, value in policy.state_dict().items()} for policy in model]
        from methods.joint_delay_ppo.learning import make_optimizers, update
        update(model, make_optimizers(model, CONFIG), [trajectory], CONFIG)
        for owner in set(range(count)) - active:
            self.assertTrue(all(torch.equal(value, model[owner].state_dict()[key])
                                for key, value in before[owner].items()))


if __name__ == '__main__':
    unittest.main()
