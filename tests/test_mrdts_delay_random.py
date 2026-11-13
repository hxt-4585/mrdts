import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from methods.mrdts_delay.high import DelayPlanner
from methods.mrdts_delay.runtime import execution_order
from methods.mrdts_delay_random.high import FlightController
from methods.mrdts_delay_random.rollout import environment, environment_snapshot, random_schedule
from methods.mrdts_delay_random.run import evaluate, load_config, predictor, train


class RandomOffloadingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config['train'].update(slots=4, rollout_slots=2, users=8, dag_nodes=3,
                                    validation_interval=1, validation_epochs=[17], evaluation_slots=1)
        self.config['high'].update(hidden=8, warmup_samples=1, fit_interval=1, fit_steps=1)
        self.snapshot = environment_snapshot(self.config['train'])

    def test_same_flight_actions_before_and_after_learning(self):
        env = environment(self.snapshot)
        env.reset(epoch_index=7)
        reference = DelayPlanner(env, self.config['high'])
        reference.reset(7)
        model = predictor(self.snapshot, self.config)
        flight = FlightController(env, model, 7)
        for exploration in (1., 0., .5):
            views, scene = reference.views(env), flight.observe(env)
            expected, actual = reference.act(views, exploration), flight.act(scene, exploration)
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            targets = np.arange(len(views), dtype=float) / 10
            reference.learn([(views, expected, targets)])
            model.update([(scene.encode(torch.from_numpy(scene.xy + actual * flight.stride)), targets)])
            begin = env.begin_step(actual)
            env.submit_schedule(random_schedule(env._runtime, begin.requests, np.random.default_rng(8)))
            env.end_step()

    def test_random_draws_match_legal_candidates_and_ers_order(self):
        env = environment(self.snapshot)
        env.reset(epoch_index=12)
        begin = env.begin_step(np.zeros((env.members.member_uav_count, 2)))
        schedule = random_schedule(env._runtime, begin.requests, np.random.default_rng(21))
        self.assertEqual(list(schedule), list(execution_order(env._runtime, begin.requests)))
        requests = {request.key: request for request in begin.requests}
        rng = np.random.default_rng(21)
        for key, placement in schedule.items():
            request = requests[key.owner_member_id, key.user_id, key.dag_id]
            candidates = env._runtime.candidate_execution_nodes(request.owner_member, request.ground_device)
            self.assertEqual(placement.execution_node, candidates[rng.integers(len(candidates))])
        env.submit_schedule(schedule)
        self.assertEqual(len(env.end_step().member_rewards), len(schedule))

    def test_checkpoint_resume_and_reproducible_random_evaluation(self):
        self.config['train']['episodes'] = 2
        with tempfile.TemporaryDirectory(prefix='mrdts_random_test_') as temp:
            root = Path(temp)
            full = train(self.config, root / 'full')
            split = train(self.config, root / 'split', episodes=1)
            train(self.config, split.parent, resume=split, episodes=2)
            actual = torch.load(split, weights_only=True)
            expected = torch.load(full, weights_only=True)
            self.assertEqual(actual['high']['seen'], 8)
            for key, tensor in expected['high']['weights'].items():
                torch.testing.assert_close(actual['high']['weights'][key], tensor, rtol=0, atol=0)
            report = evaluate(split, [17], slots=2)
            self.assertEqual(report, evaluate(full, [17], slots=2))
            row = report['rows'][0]
            self.assertEqual(row['ground_placements'] + row['member_placements'] + row['bs_placements'],
                             row['member_decisions'])
            self.assertTrue((split.parent / 'best.pt').is_file())
