import csv
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def write_csv(path, fields, rows, incomplete_tail=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if incomplete_tail is not None:
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(incomplete_tail)


class ReaderTests(unittest.TestCase):
    def test_generic_loss_is_optional_but_supplied_values_must_be_finite(self):
        from visualization.readers import read_updates

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "updates.csv"
            fields = ["epoch", "stage", "update", "total_steps", "loss"]
            row = dict(epoch=1, stage="train", update=1, total_steps=8, loss=0.25)
            write_csv(path, fields, [row])
            parsed = read_updates(path)
            self.assertEqual(parsed[0].metrics, {"loss": 0.25})
            write_csv(path, fields, [{**row, "loss": "nan"}])
            with self.assertRaisesRegex(ValueError, "finite"):
                read_updates(path)

    def test_sparse_custom_loss_and_diagnostics_keep_missing_values_and_chronology(self):
        from visualization.readers import read_updates

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "updates.csv"
            fields = ["epoch", "stage", "update", "total_steps", "critic_loss", "entropy"]
            rows = [dict(epoch=1, stage="train", update=1, total_steps=8, critic_loss=.4),
                    dict(epoch=2, stage="train", update=2, total_steps=16, entropy=.3)]
            write_csv(path, fields, rows)
            parsed = read_updates(path)
            self.assertEqual([row.metrics for row in parsed],
                             [{"critic_loss": .4}, {"entropy": .3}])
            write_csv(path, fields, list(reversed(rows)))
            with self.assertRaisesRegex(ValueError, "increasing"):
                read_updates(path)

    def test_arbitrary_validation_stages_reject_backward_boundary(self):
        from visualization.readers import read_validation

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.csv"
            row = dict(epoch=0, stage="warmup", steps=2, epoch_return=-2, mean_reward=-1)
            rows = [row, {**row, "epoch": 1, "stage": "train"}, {**row, "epoch": 1}]
            write_csv(path, list(row), rows)
            with self.assertRaisesRegex(ValueError, "chronological"):
                read_validation(path)

    def test_epochs_ignore_incomplete_tail_and_keep_stages(self):
        from visualization.readers import read_epochs

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epochs.csv"
            fields = ["epoch", "stage", "stage_epoch", "steps", "total_steps",
                      "epoch_return", "mean_reward", "mean_delay_s"]
            write_csv(path, fields, [
                {"epoch": 1, "stage": "member", "stage_epoch": 1, "steps": 2,
                 "total_steps": 2, "epoch_return": -3, "mean_reward": -1.5,
                 "mean_delay_s": 4.0},
                {"epoch": 2, "stage": "master", "stage_epoch": 1, "steps": 4,
                 "total_steps": 6, "epoch_return": -4, "mean_reward": -1,
                 "mean_delay_s": 3.0},
            ], incomplete_tail="3,master,2,4")

            rows = read_epochs(path)

        self.assertEqual([row.epoch for row in rows], [1, 2])
        self.assertEqual([row.stage for row in rows], ["member", "master"])

    def test_malformed_complete_epoch_is_rejected(self):
        from visualization.readers import read_epochs

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epochs.csv"
            write_csv(path, ["epoch", "stage", "stage_epoch", "steps", "total_steps",
                             "epoch_return", "mean_reward", "mean_delay_s"], [
                {"epoch": 1, "stage": "member", "stage_epoch": 1, "steps": 2,
                 "total_steps": 2, "epoch_return": -2, "mean_reward": -1.5,
                 "mean_delay_s": 4},
            ])
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                read_epochs(path)

    def test_nonfinite_and_nonmonotonic_updates_are_rejected(self):
        from visualization.readers import read_updates

        fields = ["epoch", "stage", "update", "total_steps", "actor_loss",
                  "value_loss", "approx_kl", "entropy", "early_stop"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "updates.csv"
            write_csv(path, fields, [
                {"epoch": 1, "stage": "member", "update": 2, "total_steps": 8,
                 "actor_loss": 0.1, "value_loss": 2, "approx_kl": 0.01,
                 "entropy": 1.2, "early_stop": "0.0"},
                {"epoch": 1, "stage": "member", "update": 1, "total_steps": 9,
                 "actor_loss": 0.2, "value_loss": 1, "approx_kl": 0.02,
                 "entropy": math.nan, "early_stop": "true"},
            ])
            with self.assertRaisesRegex(ValueError, "finite|increasing"):
                read_updates(path)

    def test_validation_accepts_trainer_schema_and_stage_boundary_at_same_epoch(self):
        from visualization.readers import read_validation

        fields = ["epoch", "stage", "steps", "epoch_return", "mean_reward", "mean_delay_s"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.csv"
            write_csv(path, fields, [
                {"epoch": 0, "stage": "member", "steps": 3, "epoch_return": -6,
                 "mean_reward": -2, "mean_delay_s": 5},
                {"epoch": 2, "stage": "member", "steps": 3, "epoch_return": -3,
                 "mean_reward": -1, "mean_delay_s": 3},
                {"epoch": 2, "stage": "master", "steps": 3, "epoch_return": -2.4,
                 "mean_reward": -.8, "mean_delay_s": 2},
            ])
            rows = read_validation(path)
        self.assertEqual([(row.epoch, row.stage) for row in rows],
                         [(0, "member"), (2, "member"), (2, "master")])

    def test_validation_rejects_duplicate_key_within_stage(self):
        from visualization.readers import read_validation

        fields = ["epoch", "stage", "steps", "epoch_return", "mean_reward", "mean_delay_s"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.csv"
            row = {"epoch": 1, "stage": "member", "steps": 2, "epoch_return": -2,
                   "mean_reward": -1, "mean_delay_s": 3}
            write_csv(path, fields, [row, row])
            with self.assertRaisesRegex(ValueError, "chronological"):
                read_validation(path)

    def test_missing_runtime_logs_are_pending_but_offline_requires_epochs(self):
        from visualization.readers import read_run

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            data = read_run(run, allow_pending=True)
            self.assertEqual(data.epochs, [])
            with self.assertRaisesRegex(ValueError, "No completed epochs"):
                read_run(run, allow_pending=False)


class FigureTests(unittest.TestCase):
    def test_single_stage_loss_only_exports_no_fabricated_diagnostics(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from visualization.readers import read_run
        from visualization.training import export, losses_figure, reward_figure
        from visualization.watch import _draw_losses

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            epoch = dict(epoch=1, stage="train", stage_epoch=1, steps=2,
                         total_steps=2, epoch_return=-4, mean_reward=-2)
            write_csv(run / "training" / "epochs.csv", list(epoch), [epoch])
            update = dict(epoch=1, stage="train", update=1, total_steps=2, loss=.75)
            write_csv(run / "training" / "updates.csv", list(update), [update])
            data = read_run(run)
            reward = reward_figure(data)
            losses = losses_figure(data)
            self.assertEqual(len(reward.axes), 2)
            self.assertEqual(len(losses.axes), 1)
            self.assertEqual(list(losses.axes[0].lines[0].get_ydata()), [.75])
            live, axes = plt.subplots(1, 2)
            _draw_losses(live, axes, data)
            self.assertEqual(len(live.axes), 1)
            self.assertEqual(list(live.axes[0].lines[0].get_ydata()), [.75])
            output = export(run)
            self.assertTrue((output / "losses.png").exists())
            self.assertFalse((output / "diagnostics.png").exists())
            with (output / "source.csv").open(encoding="utf-8") as stream:
                source = list(csv.DictReader(stream))
            self.assertEqual(source[-1]["loss"], "0.75")
            self.assertNotIn("approx_kl", source[-1])
            plt.close("all")

    def make_run(self, directory):
        run = Path(directory) / "run"
        training = run / "training"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({"experiment": {"seed": 7}}),
                                         encoding="utf-8")
        (run / "metadata.json").write_text(json.dumps({"mode": "train", "status": "complete"}),
                                           encoding="utf-8")
        epoch_fields = ["epoch", "stage", "stage_epoch", "steps", "total_steps",
                        "epoch_return", "mean_reward", "mean_delay_s"]
        write_csv(training / "epochs.csv", epoch_fields, [
            {"epoch": 1, "stage": "member", "stage_epoch": 1, "steps": 2,
             "total_steps": 2, "epoch_return": -4, "mean_reward": -2, "mean_delay_s": 5},
            {"epoch": 2, "stage": "member", "stage_epoch": 2, "steps": 2,
             "total_steps": 4, "epoch_return": -3, "mean_reward": -1.5, "mean_delay_s": 4},
            {"epoch": 3, "stage": "master", "stage_epoch": 1, "steps": 2,
             "total_steps": 6, "epoch_return": -2, "mean_reward": -1, "mean_delay_s": 3},
        ])
        update_fields = ["epoch", "stage", "update", "total_steps", "actor_loss",
                         "value_loss", "approx_kl", "entropy", "early_stop"]
        write_csv(training / "updates.csv", update_fields, [
            {"epoch": 1, "stage": "member", "update": 1, "total_steps": 2,
             "actor_loss": -.1, "value_loss": 2, "approx_kl": .01,
             "entropy": 1.2, "early_stop": "false"},
            {"epoch": 3, "stage": "master", "update": 2, "total_steps": 6,
             "actor_loss": -.05, "value_loss": 1, "approx_kl": .02,
             "entropy": .8, "early_stop": "true"},
        ])
        validation_fields = ["epoch", "stage", "steps", "epoch_return", "mean_reward", "mean_delay_s"]
        write_csv(training / "validation.csv", validation_fields, [
            {"epoch": 0, "stage": "member", "steps": 3,
             "epoch_return": -7.5, "mean_reward": -2.5, "mean_delay_s": 6},
            {"epoch": 2, "stage": "member", "steps": 3,
             "epoch_return": -3, "mean_reward": -1, "mean_delay_s": 3},
            {"epoch": 2, "stage": "master", "steps": 3,
             "epoch_return": -2.4, "mean_reward": -.8, "mean_delay_s": 2.5},
        ])
        return run

    def test_offline_command_exports_all_formats_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            output = Path(directory) / "figures"
            result = subprocess.run(
                [str(PYTHON), "-B", "-m", "visualization.training", "--run", str(run),
                 "--output", str(output)], cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for stem in ("epoch-reward", "losses", "diagnostics"):
                for suffix in ("png", "svg", "pdf"):
                    path = output / f"{stem}.{suffix}"
                    self.assertTrue(path.exists() and path.stat().st_size > 0, path)
            with (output / "source.csv").open(encoding="utf-8") as stream:
                source = list(csv.DictReader(stream))
            self.assertEqual({row["record_type"] for row in source},
                             {"epoch", "update", "validation"})

    def test_stage_smoothing_restarts_at_boundary_and_retains_raw_values(self):
        from visualization.training import stage_smooth

        smoothed = stage_smooth(["member", "member", "master"], [1.0, 3.0, 100.0], window=5)
        self.assertEqual(smoothed, [1.0, 2.0, 100.0])

    def test_recurring_stage_does_not_connect_or_smooth_across_intervening_stage(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from types import SimpleNamespace
        from visualization.training import _plot_stage_series

        stages = ["train"] * 3 + ["refine"] + ["train"] * 3
        rows = [SimpleNamespace(epoch=index, stage=stage, mean_reward=float(index))
                for index, stage in enumerate(stages, 1)]
        figure, axis = plt.subplots()
        _plot_stage_series(axis, rows, "mean_reward")
        self.assertEqual(len(axis.lines), 3)
        self.assertFalse(any("smoothed" in line.get_label() for line in axis.lines))
        self.assertEqual(list(axis.lines[-1].get_xdata()), [5, 6, 7])
        plt.close(figure)

    def test_smoothed_line_starts_only_after_a_full_five_point_window(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from types import SimpleNamespace
        from visualization.training import _plot_stage_series

        rows = [SimpleNamespace(epoch=index, stage="member", mean_reward=float(index))
                for index in range(1, 6)]
        figure, axis = plt.subplots()
        _plot_stage_series(axis, rows, "mean_reward")
        smoothed = next(line for line in axis.lines if "smoothed" in line.get_label())
        self.assertEqual(list(smoothed.get_xdata()), [5])
        self.assertEqual(list(smoothed.get_ydata()), [3.0])
        plt.close(figure)

    def test_figures_mark_stage_switch_and_do_not_smooth_short_series(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from visualization.readers import read_run
        from visualization.training import losses_figure, reward_figure

        with tempfile.TemporaryDirectory() as directory:
            data = read_run(self.make_run(directory))
            reward = reward_figure(data)
            losses = losses_figure(data)
            reward_labels = [line.get_label() for line in reward.axes[0].lines]
            loss_labels = [line.get_label() for line in losses.axes[0].lines]
            self.assertFalse(any("smoothed" in label for label in reward_labels + loss_labels))
            self.assertTrue(any(line.get_linestyle() == ":" for line in reward.axes[0].lines))
            self.assertTrue(any(line.get_linestyle() == ":" for line in losses.axes[0].lines))
            # Validation returns use a different three-step horizon than two-step training.
            self.assertIn("Validation", [line.get_label() for line in reward.axes[0].lines])
            self.assertNotIn("Validation", [line.get_label() for line in reward.axes[1].lines])
            plt.close(reward)
            plt.close(losses)

    def test_watch_once_is_headless_and_exports_reward_and_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            result = subprocess.run(
                [str(PYTHON), "-B", "-m", "visualization.watch", "--run", str(run), "--once"],
                cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((run / "figures" / "training" / "live-reward.png").exists())
            self.assertTrue((run / "figures" / "training" / "live-losses.png").exists())


if __name__ == "__main__":
    unittest.main()
