"""Member 时延奖励的手算与异步事件验证。"""

import unittest

from env.contracts import PlacementDecision
from methods.scheduling import run_online_schedule
from tests.test_serial_runtime import make_runtime, request


class TestMemberReward(unittest.TestCase):
    def test_local_snapshot_fixed_denominator_and_mean_delay(self):
        runtime, members, grounds, _ = make_runtime(duration=1., slot_start=7.)
        a = request(members[0], grounds[0], inputs=(0., 0.), cycles=(10., 10.))
        b = request(members[0], grounds[1], dag_id=1)
        other = request(members[1], grounds[2], cycles=(100.,))
        runtime.register_dags([a, b, other])
        run_online_schedule(runtime, {members[0]: (*a.task_keys, *b.task_keys),
                                      members[1]: other.task_keys},
                            lambda key: PlacementDecision(grounds[key.user_id]))
        rewards = runtime.pop_member_rewards()
        self.assertEqual([rewards[key] for key in (*a.task_keys, *b.task_keys)], [-1., -1., -.5])
        self.assertEqual(rewards[other.task_keys[0]], -10.)
        self.assertEqual(runtime.trace(b.task_keys[0]).unfinished_dags_at_decision, 1)
        self.assertEqual(runtime.trace(b.task_keys[0]).managed_dag_count, 2)
        local_mean = sum(runtime.dag_runtimes[r.key].completion_time - runtime.slot_start
                         for r in (a, b)) / 2
        self.assertEqual(sum(rewards[k] for k in (*a.task_keys, *b.task_keys)), -local_mean)
        self.assertEqual(runtime.pop_member_rewards(), {})
        runtime.finish_slot()
        self.assertEqual(runtime.pop_member_rewards(), {})

    def test_rewards_wait_for_actual_finish_and_include_queue_and_transfer(self):
        runtime, members, grounds, bs = make_runtime(duration=2.)
        slow = request(members[0], grounds[0], cycles=(30.,))
        fast = request(members[1], grounds[1], inputs=(0., 0.), cycles=(10., 10.))
        runtime.register_dags([slow, fast])
        runtime.submit_placements({slow.task_keys[0]: PlacementDecision(bs),
                                   fast.task_keys[0]: PlacementDecision(grounds[1])})
        self.assertEqual(runtime.pop_member_rewards(), {})
        runtime.advance_to_decision()
        self.assertEqual(runtime.pop_member_rewards(), {fast.task_keys[0]: -.5})
        runtime.submit_placements({fast.task_keys[1]: PlacementDecision(bs)})
        runtime.advance_to_decision()
        rewards = runtime.pop_member_rewards()
        self.assertEqual(rewards, {slow.task_keys[0]: -1.5, fast.task_keys[1]: -1.5})
        self.assertEqual(runtime.trace(fast.task_keys[1]).compute_start_at, 3.)

        runtime, members, grounds, bs = make_runtime(duration=2.)
        dag = request(members[0], grounds[0], inputs=(2.,))
        runtime.register_dags([dag])
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(bs)})
        runtime.finish_slot()
        self.assertEqual(runtime.pop_member_rewards(), {dag.task_keys[0]: -2.5})

    def test_zero_duration_and_members_without_dags(self):
        runtime, members, grounds, _ = make_runtime()
        self.assertEqual(runtime.pop_member_rewards(), {})
        dag = request(members[0], grounds[0], cycles=(0.,))
        runtime.register_dags([dag])
        runtime.submit_placements({dag.task_keys[0]: PlacementDecision(grounds[0])})
        self.assertEqual(runtime.pop_member_rewards(), {dag.task_keys[0]: 0.})
        self.assertEqual(runtime.pop_member_rewards(), {})


if __name__ == '__main__':
    unittest.main()
