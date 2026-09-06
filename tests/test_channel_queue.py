"""独立有向信道队列的单元测试。"""

import unittest

from env.runtime.channel_queue import DirectedChannelState, TransferJob
from env.types import DirectedChannelKey, EntityKind, EntityRef


class TestDirectedChannelState(unittest.TestCase):
    """同一有向实体对串行，其他有向实体对彼此独立。"""

    @staticmethod
    def _member(index: int) -> EntityRef:
        return EntityRef(EntityKind.MEMBER_UAV, index)

    @staticmethod
    def _bs() -> EntityRef:
        return EntityRef(EntityKind.BS, 0)

    @staticmethod
    def _job(
        transfer_id: str, priority_seq: int, source_ready_at: float, duration_s: float
    ) -> TransferJob:
        return TransferJob(
            transfer_id=transfer_id,
            priority_seq=priority_seq,
            source_ready_at=source_ready_at,
            duration_s=duration_s,
        )

    def test_opposite_directions_can_start_at_the_same_time(self):
        """FDD 的反向信道是独立资源，允许同刻启动。"""
        member = self._member(0)
        bs = self._bs()
        uplink = DirectedChannelState(DirectedChannelKey(member, bs))
        downlink = DirectedChannelState(DirectedChannelKey(bs, member))
        uplink.enqueue(self._job("up", 0, 0.0, 2.0))
        downlink.enqueue(self._job("down", 0, 0.0, 2.0))

        up = uplink.start_head(0.0)
        down = downlink.start_head(0.0)

        self.assertEqual(up.start_at, 0.0)
        self.assertEqual(down.start_at, 0.0)

    def test_different_entity_pairs_can_start_at_the_same_time(self):
        """两个不同 Member 到同一 BS 的链路也应相互独立。"""
        bs = self._bs()
        member_zero = DirectedChannelState(DirectedChannelKey(self._member(0), bs))
        member_one = DirectedChannelState(DirectedChannelKey(self._member(1), bs))
        member_zero.enqueue(self._job("m0-to-bs", 0, 0.0, 3.0))
        member_one.enqueue(self._job("m1-to-bs", 0, 0.0, 3.0))

        first = member_zero.start_head(0.0)
        second = member_one.start_head(0.0)

        self.assertEqual(first.start_at, 0.0)
        self.assertEqual(second.start_at, 0.0)

    def test_same_direction_starts_next_ers_job_only_after_active_job_finishes(self):
        """同一有向信道上的传输必须按照 ERS 顺序串行。"""
        channel = DirectedChannelState(DirectedChannelKey(self._member(0), self._bs()))
        channel.enqueue(self._job("first", 0, 0.0, 2.0))
        channel.enqueue(self._job("second", 1, 0.0, 1.0))

        first = channel.start_head(0.0)
        channel.finish_active(first.finish_at)
        second = channel.start_head(2.0)

        self.assertEqual(first.finish_at, 2.0)
        self.assertEqual(second.start_at, 2.0)
        self.assertEqual(second.finish_at, 3.0)

    def test_ready_later_job_cannot_bypass_unready_ers_head(self):
        """未就绪的 ERS 队首会阻塞同信道之后已就绪的作业。"""
        channel = DirectedChannelState(DirectedChannelKey(self._member(0), self._bs()))
        channel.enqueue(self._job("first", 0, 5.0, 1.0))
        channel.enqueue(self._job("second", 1, 0.0, 1.0))

        started = channel.start_head(0.0)

        self.assertIsNone(started)


if __name__ == "__main__":
    unittest.main(verbosity=2)
