# Live epoch reward implementation plan

**Goal:** Monitor the existing full-step epoch trainer without modifying existing files.

**Architecture:** A separate Python/Matplotlib process polls `epochs.csv` every two seconds. Plot completed epoch mean reward and total return; validation reward remains a distinct series. Use newline-terminated records only so concurrent appends cannot appear as completed epochs. Reload the small epoch file so resume reconciliation is reflected. No TensorBoard dependency or training hooks.

**Scope:** Continue the approved reward plot and 500-step epoch workflow. Add `scripts/watch_rl_reward.py`, `tests/test_rl_reward_watch.py`, and `methods/rl_baseline/LIVE_REWARD.md`. Preserve the current branch and existing files. Do not start a 500-epoch training job on the user's behalf.

**Verification:** Test missing logs, a partially appended record becoming complete, log truncation after resume, invalid numeric data, and headless PNG export using real saved training logs. Exercise the Tk event loop with a scheduled close. Inspect the exported figure.

**Commands:** Run targeted tests with `.venv/Scripts/python.exe -m unittest discover -s tests -p test_rl_reward_watch.py -v`; use `--once` for headless export. Normal operation uses TkAgg and exits independently of training.

**Usage:** 400 Member epochs plus 100 Master epochs gives 500 epochs, each containing 500 complete environment steps. The split is a starting example, not a validated optimal allocation. Initial and periodic validation can delay the first/next plotted point. The watcher observes logged results, not checkpoint durability or convergence.

**Verification results:** Three targeted tests passed, including append-in-progress and reading replacement logs. A real TkAgg event loop read epoch 1, received an appended epoch 2, refreshed the plot, exported PNG and closed on schedule; its temporary fixture data are only a UI test. Headless export using the existing `rl-epochs-500-verified` logs produced `live-reward.png`; visual inspection confirmed mean/return values, validation legend and stage boundary. No additional training was run. All delivered changes are new files.

**Windows resume limitation:** Independent review and local reproduction found that holding a read handle can block the trainer's `Path.replace` during resume, even with Windows DELETE sharing enabled. To preserve the user's add-only constraint, the monitor and usage guide explicitly require closing the monitor before resume and reopening after training steps restart. Normal training append monitoring is supported; concurrent resume replacement is not claimed. A trainer-side bounded retry would be a separate change to an existing file.
