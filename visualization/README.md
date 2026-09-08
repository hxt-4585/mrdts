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

The code directory `visualization/` contains no generated figures by default. For PPO,
`<run>` is `results/ppo_delay/runs/ppo_delay_ers_ppo_ppo/seed_<seed>/<timestamp_ID>/`.

| Input file under `<run>/training/` | Generated figure under `<run>/figures/training/` |
|---|---|
| `epochs.csv` and optional `validation.csv` | `epoch-reward.png/svg/pdf`: mean reward, epoch return and mean delay |
| `updates.csv` | `losses.png/svg/pdf`: actor and value loss |
| `updates.csv` | `diagnostics.png/svg/pdf`: KL and entropy |

`config.json` and `metadata.json` provide seed and run labels. `source.csv` exports the
plotted records; it is an output, not the input used to train or generate the plots.

Open an independent live reward and loss monitor:

```powershell
.venv\Scripts\python.exe -m visualization.watch --run results/<run> --interval 2
```

Use `--once` on headless systems. It writes `live-reward.png` and `live-losses.png` to the
default training figure directory. Missing logs and incomplete trailing CSV lines are treated
as pending writes by the monitor. Offline export exits with a clear error until at least one
completed epoch exists. Complete malformed rows, non-finite values, out-of-order keys, and
inconsistent epoch reward totals are rejected.
