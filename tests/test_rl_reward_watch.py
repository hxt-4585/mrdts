"""Live readers must tolerate append-in-progress and resume log replacement."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


class TestRewardWatch(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/watch_rl_reward.py'
        self.assertTrue(path.exists(), 'Standalone live reward watcher is required')
        spec = importlib.util.spec_from_file_location('reward_watch', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_missing_partial_append_and_resume_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'epochs.csv'
            self.assertEqual(self.module.read_epochs(path), [])
            header = 'epoch,stage,steps,mean_reward,epoch_return,validation_mean_reward\n'
            first = '1,member,500,-0.5,-250,\n'
            path.write_text(header + first + '2,master,500,-0.4,-200,')
            rows = self.module.read_epochs(path)
            self.assertEqual(len(rows), 1)
            with path.open('a') as file:
                file.write('-0.45\n')
            rows = self.module.read_epochs(path)
            self.assertEqual([row['epoch'] for row in rows], [1, 2])
            self.assertEqual(rows[1]['validation_mean_reward'], -.45)
            path.write_text(header + first)
            self.assertEqual(len(self.module.read_epochs(path)), 1)

    def test_inconsistent_and_nonfinite_values_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'epochs.csv'
            header = 'epoch,stage,steps,mean_reward,epoch_return\n'
            for record in ('1,member,500,-0.5,-200\n', '1,member,500,nan,nan\n'):
                path.write_text(header + record)
                with self.assertRaises(ValueError):
                    self.module.read_epochs(path)

    def test_plot_updates_existing_figure_and_exports(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'epochs.csv'
            path.write_text('epoch,stage,steps,mean_reward,epoch_return,validation_mean_reward\n'
                            '1,member,500,-0.5,-250,\n2,master,500,-0.4,-200,-0.45\n')
            figure, axes = plt.subplots(1, 2)
            status = self.module.draw(figure, axes, self.module.read_epochs(path))
            self.assertIn('2', status)
            self.assertEqual(list(axes[0].lines[0].get_ydata()), [-.5, -.4])
            self.assertEqual(list(axes[1].lines[0].get_ydata()), [-250., -200.])
            self.module.draw(figure, axes, [])
            self.assertEqual(len(axes[0].lines), 0)
            figure.savefig(Path(directory) / 'reward.png')
            self.assertGreater((Path(directory) / 'reward.png').stat().st_size, 1000)
            plt.close(figure)


if __name__ == '__main__':
    unittest.main()
