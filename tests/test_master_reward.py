"""全系统 DAG 平均完成时延奖励。"""

import unittest

import numpy as np

from env.contracts import EndStepResult
from env.runtime.slot_result import DAGResult, SlotResult
from env.step.reward import MasterReward


class TestMasterReward(unittest.TestCase):
    def test_average_is_over_dags_not_member_means_or_slot_end(self):
        slot = SlotResult(10., 30., (
            DAGResult((0, 0, 0), 11., False),
            DAGResult((0, 1, 1), 13., False),
            DAGResult((1, 2, 2), 18., False),
        ), 100., 200., ())
        self.assertEqual(MasterReward(2.).compute(slot), -2.)
        self.assertEqual(MasterReward(4.).compute(slot), -1.)
        zeros = np.zeros(2)
        result = EndStepResult(slot, zeros, zeros, zeros, zeros, zeros, zeros, False,
                               scheduling_duration_s=2.)
        self.assertEqual(result.master_reward, -2.)

    def test_zero_delay(self):
        slot = SlotResult(7., 8., (DAGResult((0, 0, 0), 7., False),), 0., 0., ())
        self.assertEqual(MasterReward(2.).compute(slot), 0.)


if __name__ == '__main__':
    unittest.main()
