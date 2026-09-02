"""可视化必须保留真实路由、并发区间与截止失败语义。"""

import importlib.util
import unittest


class TestSchedulingReplay(unittest.TestCase):
    def build(self, **kwargs):
        self.assertIsNotNone(
            importlib.util.find_spec("scripts.visualize_scheduling"),
            "尚未实现事件回放脚本",
        )
        from scripts.visualize_scheduling import build_replay
        return build_replay(**kwargs)

    def test_small_scene_and_real_ground_routing(self):
        data = self.build()
        self.assertEqual(len(data["dags"]), 4)
        self.assertEqual(len(data["tasks"]), 16)
        self.assertEqual(data["summary"]["finished_dags"], 4)
        self.assertEqual(data["summary"]["failed_dags"], 0)
        for region in range(1, 5):
            entities = [e for e in data["entities"] if e["region"] == region]
            self.assertEqual(sum(e["kind"] == "ground_device" for e in entities), 1)
            self.assertEqual(sum(e["kind"] == "member_uav" for e in entities), 2)
            tasks = [t for t in data["tasks"] if t["region"] == region]
            ground = next(e for e in entities if e["kind"] == "ground_device")
            self.assertEqual(ground["frequencies_hz"], [1e9])
            local = next(t for t in tasks if t["node"] == 1)
            self.assertEqual(local["input_arrival"], 0.0)
            self.assertEqual(local["compute_energy_j"], 0.0)
            self.assertAlmostEqual(local["finish"] - local["start"], local["cycles"] / 1e9)
            self.assertFalse(any(x["task"] == local["id"] and x["kind"] == "input"
                                 for x in data["transfers"]))
            dag = next(d for d in data["dags"] if d["region"] == region)
            ground_result = sorted(
                (x for x in data["transfers"] if x["task"] == tasks[3]["id"]
                 and x["predecessor"] == local["id"]), key=lambda x: x["hop"])
            self.assertEqual([(x["source"], x["target"]) for x in ground_result],
                             [(ground["id"], dag["owner"]), (dag["owner"], "bs")])

    def test_transfer_intervals_obey_real_source_readiness_and_serial_channels(self):
        data = self.build()
        channels = {}
        for x in data["transfers"]:
            self.assertTrue(x["completed"])
            self.assertLessEqual(x["source_ready"], x["start"])
            self.assertAlmostEqual(x["finish"] - x["start"], x["duration"])
            channels.setdefault((x["source"], x["target"]), []).append(x)
        self.assertTrue(any(x["start"] > x["source_ready"] for x in data["transfers"]))
        for jobs in channels.values():
            jobs.sort(key=lambda x: x["queue_order"])
            for before, after in zip(jobs, jobs[1:]):
                self.assertLessEqual(before["finish"], after["start"])
        self.assertAlmostEqual(sum(x["energy_j"] for x in data["transfers"]),
                               data["summary"]["tx_energy_j"])

    def test_overload_does_not_report_planned_completion_as_actual_completion(self):
        data = self.build(load_scale=10.0)
        deadline = data["meta"]["deadline"]
        self.assertGreater(data["summary"]["failed_dags"], 0)
        truncated = [x for x in data["transfers"] if x["start"] is not None and not x["completed"]]
        self.assertTrue(truncated)
        for x in truncated:
            self.assertIsNone(x["finish"])
            self.assertEqual(x["display_end"], deadline)
            self.assertFalse(any(e["kind"] == "transfer_finish" and e.get("transfer") == x["id"]
                                 for e in data["events"]))
        self.assertTrue(all(e["at"] <= deadline for e in data["events"]))
        self.assertTrue(any(t["failed_at"] == deadline for t in data["tasks"]))
        self.assertAlmostEqual(sum(x["energy_j"] for x in data["transfers"]),
                               data["summary"]["tx_energy_j"])


if __name__ == "__main__":
    unittest.main()
