# Training visualization

This package reads training logs without importing any trainer, so plots and monitoring can
run in a separate process while training writes CSV files.

Export publication-friendly offline figures:

```powershell
.venv\Scripts\python.exe -m visualization.training --run results/<run>
```

The default output is `<run>/figures/training/`. Reward, loss, and available diagnostic figures
are written as PNG, SVG, and PDF, with every plotted input row in `source.csv`. Raw training
points remain visible; stages are discovered from the logs, and the trailing mean restarts
at each stage boundary, including when a previous stage recurs.
Validation points are shown separately. Value loss uses a symlog scale (linear within
±0.0001) so different value-loss scales remain visible.
Use `--output <directory>` to choose another target.

The code directory `visualization/` contains no generated figures by default. For PPO,
`<run>` is `results/ppo/runs/ppo_ers_ppo_ppo/seed_<seed>/<timestamp_ID>/`.

| Input file under `<run>/training/` | Generated figure under `<run>/figures/training/` |
|---|---|
| `epochs.csv` and optional `validation.csv` | `epoch-reward.png/svg/pdf`: mean reward, epoch return and optional mean delay |
| Optional `updates.csv` | `losses.png/svg/pdf`: available `loss` and `*_loss` fields |
| Optional `updates.csv` | `diagnostics.png/svg/pdf`: available `approx_kl` and `entropy` fields |

An offline loss or diagnostic figure is generated only when the corresponding data exists.
Missing optional metrics remain missing; the reader never fills them with zero. The live
monitor discovers new loss fields as rows arrive and shows a waiting message before loss
data is available.

The common CSV contract is:

- `epochs.csv`: required `epoch`, `stage`, `stage_epoch`, `steps`, `total_steps`,
  `epoch_return`, `mean_reward`; optional `mean_delay_s`.
- `validation.csv`: required `epoch`, `stage`, `steps`, `epoch_return`, `mean_reward`;
  optional `mean_delay_s`. Validation may be absent.
- `updates.csv`: required `epoch`, `stage`, `update`, `total_steps`; optional numeric
  `loss`, any name ending in `_loss`, `approx_kl`, `entropy`, and boolean `early_stop`.
  Blank optional cells are skipped; supplied numeric metrics must be finite.

Stage names are arbitrary nonempty strings. Epochs increase strictly; validation keys
are ordered by epoch and stage first appearance, allowing the end of one stage and the
start of another at the same epoch. Update keys increase by `(epoch, update)`, and logged
`total_steps` cannot decrease. The existing two-stage PPO logs satisfy this same contract.

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
