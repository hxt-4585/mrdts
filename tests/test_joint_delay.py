import ast
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('joint_delay_ppo', 'joint_delay_acac')


class JointDelayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_high_level_starts_identically_and_excludes_hidden_information(self):
        controllers = []
        for method in METHODS:
            support = importlib.import_module(f'methods.{method}.support')
            planner = importlib.import_module(f'methods.{method}.planner')
            config = support.load_config()
            env = support.make_environment(support.environment_snapshot(config))
            frames = planner.observe(env)
            controller = planner.Controller(frames, config['high'])
            controllers.append(controller)
            for frame in frames:
                np.testing.assert_array_equal(frame.positions[~frame.visible], 0)
            controller.reset(400000)
            first = controller.actions(frames, 1.)
            controller.reset(400000)
            np.testing.assert_array_equal(first, controller.actions(frames, 1.))
        for a, b in zip(controllers[0].models, controllers[1].models):
            for key, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def test_proxy_updates_and_flight_remains_feasible(self):
        for method in METHODS:
            support = importlib.import_module(f'methods.{method}.support')
            planner = importlib.import_module(f'methods.{method}.planner')
            config = support.load_config()
            config['high'].update(hidden=16, warmup_samples=2, fit_interval=1, replay_size=8)
            env = support.make_environment(support.environment_snapshot(config))
            frames = planner.observe(env)
            controller = planner.Controller(frames, config['high'])
            before = [p.detach().clone() for p in controller.models[0].parameters()]
            for _ in range(3):
                actions = controller.actions(frames, 0.)
                controller.feedback([controller.experience(frames, actions, np.ones(len(frames)))])
            self.assertTrue(any(not torch.equal(p, q) for p, q in zip(before, controller.models[0].parameters())))
            actions = controller.actions(frames, 0.)
            self.assertLessEqual(np.abs(actions).max(), 1.)
            count = env.members.member_uav_count
            end = env.members.positions[:count, :2] / env.region.config.side_length + actions * frames[0].step
            self.assertTrue(np.all((end >= 0) & (end <= 1)))

    def test_baselines_are_independent(self):
        for method in METHODS:
            for path in (ROOT / 'methods' / method).glob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('methods.'):
                        self.assertTrue(node.module.startswith(f'methods.{method}.'), (path, node.module))

    def test_sampler_budget_drains_when_workers_outnumber_rollout_slots(self):
        for method in METHODS:
            run = importlib.import_module(f'methods.{method}.run')
            remaining = [3] * 9
            for iteration in range(14):
                quotas = run.slot_quotas(remaining, 2, iteration)
                self.assertEqual(sum(quotas), min(2, sum(remaining)))
                self.assertTrue(all(0 <= q <= r for q, r in zip(quotas, remaining)))
                remaining = [r - q for r, q in zip(remaining, quotas)]
            self.assertEqual(remaining, [0] * 9)

    def test_planner_matches_distributed_delay_math_and_ers_matches_between_groups(self):
        from methods.distributed_delay.planner import Frame, LocalProxy, coordinate
        from methods.joint_delay_ppo.planner import Controller, observe, optimize
        from methods.joint_delay_ppo.support import load_config, make_environment, environment_snapshot
        from methods.joint_delay_ppo.rollout import rank_orders
        from methods.joint_delay_acac.rollout import order_requests
        config = load_config()
        env = make_environment(environment_snapshot(config))
        views = observe(env)
        controller = Controller(views, config['high'])
        original_views, original_models = [], []
        for view, model in zip(views, controller.models):
            original_view = Frame(view.positions, view.visible, view.owned, view.users, view.neighbors, view.step)
            original = LocalProxy(len(view.positions), view.users)
            original.network.load_state_dict(model.regressor.state_dict())
            point = torch.tensor(view.positions)
            torch.testing.assert_close(model.encode(view, point), original.features(point, view.visible, view.owned))
            original_views.append(original_view)
            original_models.append(original)
        initial = np.zeros_like(views[0].positions)
        expected, _ = coordinate(original_views, original_models, initial,
                                 rounds=config['high']['rounds'], rate=config['high']['gradient_rate'])
        np.testing.assert_allclose(optimize(views, controller.models, initial, config['high']), expected,
                                   rtol=1e-5, atol=1e-6)
        for epoch in (0, 17, 400000):
            env.reset(epoch_index=epoch)
            begin = env.begin_step(initial)
            self.assertEqual(rank_orders(env._runtime, begin.requests), order_requests(env._runtime, begin.requests))

    def test_training_updates_both_levels_and_frozen_evaluation_is_repeatable(self):
        for method in METHODS:
            support = importlib.import_module(f'methods.{method}.support')
            run = importlib.import_module(f'methods.{method}.run')
            config = support.load_config()
            config['train'].update(episodes=2, slots=4, workers=2, rollout_slots=4,
                                   hidden=16, ppo_epochs=1, device='cpu', users=4, dag_nodes=3,
                                   validation_interval=2, validation_epochs=[1500000],
                                   test_epochs=[2500000], evaluation_slots=2, checkpoint_interval=1)
            config['high'].update(hidden=16, replay_size=16, warmup_samples=1, fit_interval=1)
            with tempfile.TemporaryDirectory(prefix='joint_delay_smoke_') as temporary:
                output = Path(temporary) / 'run'
                run.train(config, output)
                saved = torch.load(output / 'latest.pt', map_location='cpu', weights_only=True)
                self.assertEqual(saved['episodes_completed'], 2)
                self.assertEqual(saved['slots_completed'], 8)
                self.assertGreater(saved['updates'], 0)
                self.assertTrue(all(state['samples'] == 8 for state in saved['high']))
                initial = torch.load(output / 'initial.pt', map_location='cpu', weights_only=True)
                for section in ('low',):
                    self.assertTrue(any(not torch.equal(v, saved[section][k]) for k, v in initial[section].items()))
                self.assertTrue(any(not torch.equal(v, saved['high'][0]['model'][k])
                                    for k, v in initial['high'][0]['model'].items()))
                a = run.evaluate(output / 'best.pt', [2500010], slots=2)
                b = run.evaluate(output / 'best.pt', [2500010], slots=2)
                self.assertEqual(a[0]['delay_sum_s'], b[0]['delay_sum_s'])
                self.assertEqual(a[0]['flight_violations'], 0)
                self.assertEqual(saved['config']['train']['workers'], 2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_full_size_acac_cuda_learner_with_cpu_samplers(self):
        from methods.joint_delay_acac.support import load_config
        from methods.joint_delay_acac.run import train
        config = load_config()
        config['train'].update(episodes=2, slots=8, evaluation_slots=2,
                               validation_epochs=[1500000], test_epochs=[2500000])
        with tempfile.TemporaryDirectory(prefix='joint_cuda_smoke_') as temporary:
            output = train(config, Path(temporary) / 'run')
            state = torch.load(output / 'latest.pt', map_location='cpu', weights_only=True)
            self.assertEqual(state['slots_completed'], 16)
            self.assertEqual(state['updates'], 2)
            self.assertTrue(all(t.device.type == 'cpu' for t in state['low'].values()))
            self.assertTrue(all(s['samples'] == 16 for s in state['high']))


if __name__ == '__main__':
    unittest.main()
