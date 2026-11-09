import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


class VelocityTests(unittest.TestCase):
    def test_frozen_validation_records_actual_velocity_and_preserves_checkpoint(self):
        from methods.joint_delay_acac.velocity import run
        from methods.joint_delay_acac.support import load_config, environment_snapshot, make_environment
        from methods.joint_delay_acac.learning import make_model
        from methods.joint_delay_acac.planner import Controller, observe
        torch.set_num_threads(1)
        config = load_config()
        config['train'].update(hidden=16, users=4, dag_nodes=3)
        config['high'].update(hidden=16)
        snapshot = environment_snapshot(config)
        env = make_environment(snapshot)
        count = env.members.member_uav_count
        model = make_model(count, 5 * count + 8, config['train'], 'cpu')
        high = Controller(observe(env), config['high'])
        with tempfile.TemporaryDirectory(prefix='acac_velocity_smoke_') as temporary:
            source = Path(temporary) / 'checkpoint.pt'
            torch.save(dict(config=config, environment=snapshot, low=model.state_dict(),
                            high=[m.snapshot() for m in high.models], selected_episode=0), source)
            original = hashlib.sha256(source.read_bytes()).hexdigest()
            destination = Path(temporary) / 'output'
            destination.mkdir()
            output = run(source, destination, epoch=1500000, slots=3)
            with (output / 'velocities.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3 * count)
            for row in rows:
                vx, vy = float(row['vx_mps']), float(row['vy_mps'])
                self.assertAlmostEqual(vx, (float(row['x_after_m']) - float(row['x_before_m'])) / .5)
                self.assertAlmostEqual(vy, (float(row['y_after_m']) - float(row['y_before_m'])) / .5)
                self.assertAlmostEqual(float(row['speed_mps']), np.hypot(vx, vy))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['slots'], 3)
            self.assertEqual(report['flight_violations'], 0)
            self.assertTrue((output / 'fleet_speed.png').exists())
            self.assertTrue((output / 'uav_speeds.png').exists())
            from scripts.plot_acac_trajectories import plot
            plot(output)
            trajectory = json.loads((output / 'trajectory_summary.json').read_text())
            self.assertTrue(trajectory['initial_positions_match_snapshot'])
            self.assertEqual(trajectory['slots'], 3)
            self.assertTrue((output / 'uav_trajectories.png').exists())


if __name__ == '__main__':
    unittest.main()
