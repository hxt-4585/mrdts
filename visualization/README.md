# Training visualization

This package reads PPO logs without importing the trainer, so plots and monitoring can
run in a separate process while training writes CSV files.

Export publication-friendly offline figures:

```powershell
.venv\Scripts\python.exe -m visualization.training --run results/<run>
```

The default output is `<run>/figures/training/`. Reward, loss, and PPO diagnostic figures
are written as PNG, SVG, and PDF, with every plotted input row in `source.csv`. Raw training
points remain visible; the trailing mean restarts at each Member/Master stage boundary.
Validation points are shown separately. Value loss uses a symlog scale (linear within
±0.0001) so both Member and Master losses remain visible despite different return scales.
Use `--output <directory>` to choose another target.

Open an independent live reward and loss monitor:

```powershell
.venv\Scripts\python.exe -m visualization.watch --run results/<run> --interval 2
```

Use `--once` on headless systems. It writes `live-reward.png` and `live-losses.png` to the
default training figure directory. Missing logs and incomplete trailing CSV lines are treated
as pending writes by the monitor. Offline export exits with a clear error until at least one
completed epoch exists. Complete malformed rows, non-finite values, out-of-order keys, and
inconsistent epoch reward totals are rejected.
