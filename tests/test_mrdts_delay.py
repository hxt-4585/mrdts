"""Behavioral tests for the independent delay baseline."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import unittest
import tempfile

from methods.mrdts_delay.high import DelayPlanner, regional_targets
from methods.mrdts_delay.low import PPO, Transition, advantages, member_input
from methods.mrdts_delay.runtime import build_environment, default_settings, run_slot
from methods.mrdts_delay.storage import load_config


def small_environment():
    from dataclasses import replace
    settings = default_settings()
    settings['user'] = replace(settings['user'], total_users=12, min_users_per_region=1)
    settings['dag'] = replace(settings['dag'], n=4)
    return build_environment(settings)


def test_features_contain_only_positions_visibility_and_ownership():
    env = small_environment()
    planner = DelayPlanner(env, load_config()['high'])
    view = planner.views(env)[0]
    destination = torch.tensor(view.xy, requires_grad=True)
    encoded = planner.models[0].features(view, destination)
    count = env.members.member_uav_count
    assert encoded.shape == (1, 4 * count)
    np.testing.assert_array_equal(encoded[0, :2 * count].detach(),
                                  (view.xy * view.visible[:, None]).ravel())
    np.testing.assert_array_equal(encoded[0, 2 * count:3 * count].detach(), view.visible)
    np.testing.assert_array_equal(encoded[0, 3 * count:].detach(), view.owned)
    encoded[:, :2 * count].sum().backward()
    np.testing.assert_array_equal(destination.grad, np.repeat(view.visible[:, None], 2, axis=1))


def test_regional_predictions_sum_to_global_mean_even_without_owner():
    dags = (SimpleNamespace(key=(-1, 0, 0), completion_time=14.),
            SimpleNamespace(key=(1, 1, 1), completion_time=18.),
            SimpleNamespace(key=(2, 2, 2), completion_time=22.))
    slot = SimpleNamespace(dags=dags, slot_start=10.)
    target = regional_targets(slot, np.array([1, 2, 2]), 2)
    np.testing.assert_allclose(target, [4 / 3, 20 / 3])
    assert np.isclose(target.sum(), 8.)


def test_member_input_retains_user_position_and_removes_energy():
    count = 3
    raw = np.arange(5 * count + 6, dtype=np.float64)
    actual = member_input(raw, count)
    expected = np.concatenate((raw[:4 + 4 * count], raw[4 + 5 * count:]))
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (4 * count + 6,)
    np.testing.assert_array_equal(actual[-2:], raw[-2:])


def test_gae_ends_at_slot_boundary():
    samples = [Transition(np.zeros(2), np.ones(2, bool), 0, 0., .5, -1.),
               Transition(np.zeros(2), np.ones(2, bool), 1, 0., .2, -2.)]
    advantages(samples, gamma=1., trace=1.)
    np.testing.assert_allclose([s.target for s in samples], [-3., -2.])


def test_all_local_window_returns_delay_without_member_decisions():
    env = small_environment()
    # All members are in region 1 and all users are in region 2.
    env.members.positions[:-1, :2] = env.masters.positions[0, :2]
    env.users.positions[:, :2] = env.masters.positions[1, :2]
    env.users.region_ids.fill(2)
    actions = np.zeros((env.members.member_uav_count, 2))
    trajectories, row, targets = run_slot(env, None, actions, load_config()['train'])
    assert not any(trajectories)
    assert row['member_decisions'] == 0
    assert row['member_reward_sum'] == 0
    assert row['dag_count'] == env.users.num_users
    assert row['mean_delay_s'] > 0
    assert np.isclose(targets[1], row['mean_delay_s'])


def test_real_slot_training_feedback_and_planning():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    config = load_config()
    env = small_environment()
    count = env.members.member_uav_count
    low = PPO(count, config['train'])
    planner = DelayPlanner(env, config['high'])
    before = [policy.actor[0].weight.detach().clone() for policy in low.policies]
    views = planner.views(env)
    action = planner.act(views, 0.)
    trajectories, row, targets = run_slot(env, low, action, config['train'])
    assert row['dag_count'] == 12
    assert np.isclose(targets.sum(), row['mean_delay_s'])
    assert row['flight_violations'] == 0
    assert all(np.isfinite(s.reward) for seq in trajectories for s in seq)
    low.learn([trajectories])
    assert any(not torch.equal(old, policy.actor[0].weight) for old, policy in zip(before, low.policies))
    planner.learn([(views, action, targets)])
    assert all(m.size == 1 for m in planner.models)
    planner.config = dict(planner.config, warmup_samples=1)
    action = planner.act(planner.views(env), 0.)
    assert np.isfinite(action).all()
    assert (np.abs(action) <= 1).all()
    assert row['member_reward_sum'] < 0


def test_no_other_method_dependencies():
    folder = Path(__file__).resolve().parents[1] / 'methods' / 'mrdts_delay'
    for path in folder.glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith('methods.') or node.module.startswith('methods.mrdts_delay')
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith('methods.') or a.name.startswith('methods.mrdts_delay')
                           for a in node.names)


def test_train_resume_evaluation_and_total_parallel_slot_budget():
    from methods.mrdts_delay.run import train, evaluate
    config = load_config()
    config['train'].update(episodes=1, slots=3, workers=2, rollout_slots=2, users=8,
                           dag_nodes=3, hidden=16, ppo_epochs=1, validation_interval=1,
                           validation_epochs=[1500000], evaluation_slots=1)
    config['high'].update(hidden=8, warmup_samples=1, fit_interval=1, fit_steps=1, replay_size=2)
    with tempfile.TemporaryDirectory(prefix='mrdts_delay_test_') as folder:
        root = Path(folder)
        train(config, root / 'resumed')
        first = torch.load(root / 'resumed' / 'last.pt', weights_only=True)
        assert first['completed_episodes'] == 1
        assert first['total_slots'] == 3
        assert (root / 'resumed' / 'best.pt').exists()
        train(config, root / 'resumed', resume=root / 'resumed' / 'last.pt', episodes=2)
        config['train']['episodes'] = 2
        train(config, root / 'whole')
        resumed = torch.load(root / 'resumed' / 'last.pt', weights_only=True)
        whole = torch.load(root / 'whole' / 'last.pt', weights_only=True)
        assert resumed['total_slots'] == whole['total_slots'] == 6
        for key, tensor in whole['low']['weights'].items():
            assert torch.equal(tensor, resumed['low']['weights'][key]), key
        for expected, actual in zip(whole['high']['models'], resumed['high']['models']):
            for key, tensor in expected['weights'].items():
                assert torch.equal(tensor, actual['weights'][key]), key
        frozen = evaluate(root / 'resumed' / 'last.pt', [1500000], slots=1, workers=1)
        assert frozen['summary']['slots'] == 1
        assert np.isfinite(frozen['summary']['mean_delay_s'])


def test_planning_does_not_reweight_regional_predictions_by_user_count():
    from dataclasses import replace
    env = small_environment()
    config = dict(load_config()['high'], rounds=1)
    planner = DelayPlanner(env, config)

    class LinearDelay:
        def __call__(self, view, destination):
            owned = torch.as_tensor(view.owned, dtype=destination.dtype)
            return (.01 * (destination[:, 0] * owned).sum()).reshape(1)

    planner.models = [LinearDelay() for _ in planner.models]
    views = planner.views(env)
    initial = np.zeros((env.members.member_uav_count, 2), np.float32)
    actual = planner.optimize(views, initial)
    duplicated = [replace(view, users=np.repeat(view.users, 7, axis=0)) for view in views]
    np.testing.assert_array_equal(actual, planner.optimize(duplicated, initial))
    np.testing.assert_allclose(actual[:, 0], -.01 * config['gradient_rate'] / views[0].stride,
                               rtol=1e-5)
    np.testing.assert_array_equal(actual[:, 1], 0.)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_'))
