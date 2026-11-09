import ast
from pathlib import Path
import unittest
import tempfile

import numpy as np

from methods.distributed_lyapunov.simulation import build_env, objective, queue_statistics
from methods.distributed_lyapunov.learning import MemberPolicy
from methods.distributed_lyapunov.scheduler import rank_orders
from methods.distributed_lyapunov.scheduler import route_cost
from methods.distributed_lyapunov.simulation import execute
from methods.distributed_lyapunov.planner import Controller, frames
from methods.distributed_lyapunov.run import pilot_needs_confirmation
from methods.distributed_lyapunov.run import train, save
from env.types import EntityKind, EntityRef, DirectedChannelKey
from env.communication.routing import RoutePlan
import torch


class LyapunovTests(unittest.TestCase):
    def test_nonzero_stability_excludes_initial_transient_and_linear_drift(self):
        from methods.distributed_lyapunov.nonzero import nonzero_statistics
        n = 4000
        energy = np.full((n, 2), 128.3)
        q = np.zeros((n, 2))
        q[:100, 0] = 20.
        self.assertFalse(nonzero_statistics(q, energy, 128.4)['nonzero_stable'])
        q[:, 0] = 2. + np.sin(np.arange(n)/10.)
        self.assertTrue(nonzero_statistics(q, energy, 128.4)['nonzero_stable'])
        q[:, 0] = np.arange(n)*.05
        self.assertFalse(nonzero_statistics(q, energy, 128.4)['nonzero_stable'])

    def test_high_handoff_keeps_low_weights_frozen(self):
        torch.set_num_threads(1)
        env = build_env(220.)
        policies = torch.nn.ModuleList([MemberPolicy(12) for _ in range(12)])
        controller = Controller(frames(env), 45)
        config = dict(seed=45, budget=220., v=100000., high_episodes=1, train_slots=2,
                      validation_slots=2, validation_interval=1, min_episodes=1,
                      patience=1, test_slots=2)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder/'low.pt'
            save(source, policies, controller, config, env, 10)
            train(config, folder/'high', 'staged', source)
            selected = torch.load(folder/'high'/'selected.pt', weights_only=False)
            for name, value in policies.state_dict().items():
                self.assertTrue(torch.equal(value, selected['policies'][name]))

    def test_pilot_confirms_energy_feasible_short_bursts(self):
        # A transient tail slope in 128 slots must trigger a longer run,
        # rather than declare all budgets infeasible or relax final QA.
        self.assertTrue(pilot_needs_confirmation(dict(empirically_stable=False,
                        mean_energy_j=[180., 185.]), 195.))
        self.assertFalse(pilot_needs_confirmation(dict(empirically_stable=False,
                         mean_energy_j=[180., 200.]), 195.))

    def test_objective_uses_start_queue_and_only_uav_energy(self):
        self.assertEqual(objective(3., np.array([2., 4.]), np.array([5., 7.]), 10.), 68.)

    def test_member_retains_queue_features(self):
        env = build_env(145.)
        env.reset(epoch_index=17)
        begin = env.begin_step(np.zeros((env.members.member_uav_count, 2)))
        key = next(iter(rank_orders(env, begin.requests, np.zeros(12), 1000.).values()))[0]
        env.members.energy_virtual_queue_j[:] = 290.
        obs, mask = MemberPolicy.inputs(env, key)
        self.assertEqual(len(obs), 6 * env.members.member_uav_count + 8)
        start = 4 + 4 * env.members.member_uav_count
        self.assertEqual(obs[start + key.owner_member_id], 2.)
        self.assertEqual(len(mask), env.members.member_uav_count + 2)

    def test_growth_detected_per_uav_without_averaging_it_away(self):
        q = np.column_stack((np.arange(100) * 10., np.zeros(100)))
        stats = queue_statistics(q, np.tile([155., 130.], (100, 1)), 145.)
        self.assertFalse(stats['empirically_stable'])
        self.assertAlmostEqual(stats['tail_slope_j_per_slot'][0], 10.)

    def test_uav_sender_energy_only_in_rank(self):
        env = build_env(145.)
        env.reset(epoch_index=17)
        env.begin_step(np.zeros((12, 2)))
        runtime = env._runtime
        ground = EntityRef(EntityKind.GROUND_DEVICE, 19)
        uav = EntityRef(EntityKind.MEMBER_UAV, 0)
        bs = EntityRef(EntityKind.BS, 0)
        for source, target, weighted in [(ground, uav, False), (bs, uav, False), (uav, bs, True)]:
            route = RoutePlan((DirectedChannelKey(source, target),))
            a = route_cost(runtime, route, np.zeros(12), 1000.)
            b = route_cost(runtime, route, np.ones(12)*500., 1000.)
            self.assertEqual(b > a, weighted)

    def test_real_execution_preserves_queue_identity(self):
        env = build_env(145.)
        env.reset(epoch_index=17)
        env.members.energy_virtual_queue_j[:] = np.arange(12)*20.
        before = env.members.energy_virtual_queue_j.copy()
        _, row, costs = execute(env, [None]*12, np.zeros((12, 2)), 1000., heuristic=True)
        np.testing.assert_allclose(row['queue_j'], np.maximum(before + row['energy_j'] - 145., 0.))
        self.assertAlmostEqual(costs.sum(), row['objective'])

    def test_projection_uses_learned_gradient_and_boundaries(self):
        torch.set_num_threads(1)
        env = build_env(145.)
        current = frames(env)
        controller = Controller(current, 45)
        actions, _ = controller.actions(current)
        for _ in range(64):
            controller.feedback(current, actions, np.ones(4)*1000., 1000.)
        actions, payload = controller.actions(current, exploration=.8)
        self.assertTrue(np.isfinite(actions).all())
        self.assertTrue((np.abs(actions) <= 1.).all())
        endpoint = env.members.positions[:12,:2] + actions * 5.
        self.assertTrue((endpoint >= 0.).all())
        self.assertTrue((endpoint <= env.region.config.side_length).all())
        self.assertGreater(payload, 0)

    def test_baseline_has_no_imports_from_other_methods(self):
        for path in Path('methods/distributed_lyapunov').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith('methods.') and
                                     not node.module.startswith('methods.distributed_lyapunov'))


if __name__ == '__main__':
    unittest.main()
