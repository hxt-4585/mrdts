# PPO delay baseline migration

Approved in conversation on 2026-09-08. Source: `c47059dee760e0095b7d135a856a114fa63e1a5e`.

Preserve the source's 23-feature Member candidate scorer, learnable ground bias of 4,
continuous tanh Gaussian Master, separate centralized critics, staged updates, clipped
PPO settings and finite-horizon GAE. Adopt its full physical-epoch loop: default 500
slots, Member flush every 8 slots, Master update at the true episode boundary.
Both actors act at every slot; only the current stage's actor/critic update.

Reward is negative global mean censored DAG delay divided by the scheduling window.
Failures contribute the window length. No additional energy or violation penalties.
Record epoch return as the sum of actual physical-slot rewards.

Use the current Scene, RandomStreams, Simulator, ERS and priority_seq contracts.
All payload features use decimal Kbit (1000 bits), as does the current simulator.
Keep fixed spatial layout for a seed, reset UAV positions per episode, and reserve
disjoint episode-ID ranges for validation/test/training without separate user seeds.

Generic PPO, rollout storage and MLP/value helpers belong in methods/learning.
UAV/DAG observations, actors, rewards and training lifecycle belong to
methods/solutions/ppo. Flight/scheduling wrappers belong to methods/components.
Training and deployment share method execution. Register the method and Trainer;
keep experiments.train/run as public entry points and support explicit CPU/CUDA.

Use the same results/<experiment>/runs/<method>/seed_<seed>/<timestamp_ID> layout as
Random. Common metadata/config/metrics/summary stay at the run root; training CSVs
go in training/, checkpoints in checkpoints/, plots in figures/training/.
Independent deterministic evaluation creates its own run and identifies its source
checkpoint. Comparison aggregation includes evaluation mode by default.

New sibling visualization/ owns readers, common style, offline training plots and
an independent live monitor. It reads completed CSV lines, distinguishes stages and
training/validation, retains raw data alongside within-stage smoothing, and exports
PNG/SVG/PDF. It must not fabricate history or require training to open a GUI.

Resume uses the latest complete epoch, restores optimizers and RNG, checks physical,
feature and split settings, and archives uncheckpointed log rows before replay.
Entering Master freezes the validation-selected Member permanently for that run.

Old KB results and checkpoints are historical evidence, not current Kbit results.
Migration validation includes exact probability/update parity, physical integration,
device handling, staged freeze, checkpoint round trip, uninterrupted/resumed equality,
paired evaluation, plotting and a real bounded training experiment.
