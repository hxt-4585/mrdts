"""可视化必须保留真实路由、并发区间与完整执行时间。"""

import importlib.util
import unittest

from env.settings import EnvConfig, RegionConfig, UserConfig


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
        region_count = RegionConfig.default().region_count
        ground_config = UserConfig.default()
        deadline = data["meta"]["slot_start"] + EnvConfig.default().scheduling_duration_s
        self.assertEqual(data["meta"]["deadline"], deadline)
        self.assertEqual(len(data["dags"]), region_count)
        self.assertEqual(len(data["tasks"]), 4 * region_count)
        finished_dags = 0
        for region in range(1, region_count + 1):
            entities = [e for e in data["entities"] if e["region"] == region]
            self.assertEqual(sum(e["kind"] == "ground_device" for e in entities), 1)
            self.assertEqual(sum(e["kind"] == "member_uav" for e in entities), 2)
            tasks = [t for t in data["tasks"] if t["region"] == region]
            ground = next(e for e in entities if e["kind"] == "ground_device")
            self.assertEqual(ground["frequencies_hz"], [ground_config.core_frequency] * ground_config.core_count)
            local = next(t for t in tasks if t["node"] == 1)
            self.assertIn("input_kbit", local)
            self.assertNotIn("input_kb", local)
            self.assertEqual(local["input_arrival"], local["decision_at"])
            self.assertTrue(all(t["decision_at"] == data["meta"]["slot_start"] for t in tasks))
            self.assertGreater(local["start"], local["input_arrival"])
            self.assertEqual(local["compute_energy_j"], 0.0)
            self.assertAlmostEqual(local["finish"] - local["start"], local["cycles"] / ground_config.core_frequency)
            self.assertFalse(any(x["task"] == local["id"] and x["kind"] == "input"
                                 for x in data["transfers"]))
            dag = next(d for d in data["dags"] if d["region"] == region)
            completion = max(t["finish"] for t in tasks)
            self.assertEqual(dag["completion"], completion)
            self.assertEqual(dag["failed"], completion > deadline)
            finished_dags += int(completion <= deadline)
            self.assertTrue(all("kbit" in edge and "kb" not in edge for edge in dag["edges"]))
            sink = next(t for t in tasks if t["node"] == 3)
            ground_result = sorted(
                (x for x in data["transfers"] if x["task"] == sink["id"]
                 and x["predecessor"] == local["id"]), key=lambda x: x["hop"])
            self.assertEqual([(x["source"], x["target"]) for x in ground_result],
                             [(ground["id"], dag["owner"]), (dag["owner"], "bs")])
        self.assertEqual(data["summary"]["finished_dags"], finished_dags)
        self.assertEqual(data["summary"]["failed_dags"], region_count - finished_dags)

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

    def test_replay_sequence_comes_from_actual_ers_costs(self):
        data = self.build()
        ordered = sorted(data["tasks"], key=lambda task: task["ers"])
        self.assertIn("rank_s", ordered[0])
        self.assertEqual([task["ers"] for task in ordered], list(range(len(ordered))))
        self.assertEqual([task["rank_s"] for task in ordered],
                         sorted((task["rank_s"] for task in ordered), reverse=True))
        by_id = {task["id"]: task for task in ordered}
        for task in ordered:
            tail = max((cost + by_id[f'{task["dag"]}-t{child}']["rank_s"]
                        for child, cost in task["average_edge_comm_s"].items()), default=0.)
            self.assertAlmostEqual(task["rank_s"], task["average_compute_s"] + tail)

    def test_overload_replay_includes_full_execution_beyond_nominal_boundary(self):
        data = self.build(load_scale=10000.0)
        self.assertEqual(data["summary"]["failed_dags"], len(data["dags"]))
        self.assertEqual(data["summary"]["finished_dags"], 0)
        self.assertGreater(data["meta"]["slot_end"], data["meta"]["deadline"])
        self.assertTrue(all(x["completed"] for x in data["transfers"]))
        self.assertTrue(all(t["finish"] is not None
                            for t in data["tasks"]))
        self.assertTrue(all(e["at"] <= data["meta"]["slot_end"] for e in data["events"]))
        self.assertTrue(any(e["at"] > data["meta"]["deadline"] for e in data["events"]))
        self.assertAlmostEqual(sum(x["energy_j"] for x in data["transfers"]),
                               data["summary"]["tx_energy_j"])


if __name__ == "__main__":
    unittest.main()
