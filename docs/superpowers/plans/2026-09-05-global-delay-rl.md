# Global delay RL baseline implementation plan

Goal: add runnable Master/Member PPO training without editing any pre-existing file.
User authorization: implement on a new branch, use default environment configuration,
optimize global delay only, report before any necessary edit to existing code.

Architecture: an external adapter uses Simulator.begin_slot, ERS.plan,
SchedulingRuntime.submit_dags and Simulator.end_slot. Shared Member candidate scorer
and shared masked Master Gaussian actor have separate centralized value networks.
First train Member at stationary topology; freeze it before training Master across
multiple slots. Actor inputs remain local. Existing environment files are untouched.

Objective: negative mean per-DAG completion delay / deadline; incomplete DAGs are
assigned the deadline. This is a censored delay objective, not an uncensored makespan
or an extra failure penalty. Log failure rate separately. No energy/distance reward.
Member planning steps consume zero simulation time, use undiscounted Monte Carlo
terminal returns. Master uses GAE over actual slots, with gamma=1 and finite episodes.
Invalid flights leaving an occupied region empty are rejected jointly by the adapter;
map-boundary rejection is performed by the original environment. No added penalty.

All files below are new. Execute inline; no separate task or worktree is required.

- [x] Add tests/test_rl_adapter.py: default 100 DAGs, ground execution agrees with
  sum(cycles)/ground_frequency, global reward weights DAGs, cross-region masks and
  IDs refresh, empty-region rejection, planning leaves runtime untouched.
  Run `.venv/Scripts/python.exe -m unittest tests.test_rl_adapter -v` (RED).
- [x] Add methods/rl_baseline/{__init__,adapter,observations}.py. Keep all placement
  decisions until one atomic batch submission. Rerun adapter tests (GREEN).
- [x] Add tests/test_rl_training.py: illegal categorical actions have zero
  probability, Gaussian ownership masks exclude unrelated UAVs, actual PPO updates
  change parameters with finite losses, freeze and checkpoint inference roundtrip.
  Run the new tests (RED).
- [x] Add methods/rl_baseline/{networks,ppo,runner}.py and requirements-rl.txt.
  Round-robin Member inference batches one current ERS task from each active Member;
  each Member sees only its own previous assignments. Keep stored rollout masks.
  Normalize advantages, clip PPO and gradients, stop epochs for excessive KL.
- [x] Add scripts/train_rl.py and scripts/evaluate_rl.py with seeded CPU execution,
  independent held-out DAG batches (default generator stream, disjoint offsets),
  CSV metrics, atomic checkpoints, resume and deterministic evaluation. Save resolved
  environment and training configurations alongside artifacts.
- [x] Run default-size smoke training, checkpoint reload evaluation, original and
  added tests. Run a bounded learning check and report measured improvement without
  claiming guaranteed convergence. Final implementation also completed three full seeds.
- [x] Add methods/rl_baseline/README.md and a measured verification report. Check
  `git diff HEAD --name-status` and untracked inventory prove additions only.

Artifacts default to scripts/output/rl-baseline (already ignored by existing rules).
Pytorch is an optional dependency installed into the existing virtual environment;
do not run dependency synchronization that removes it. Original dependencies/lock
files are not changed. Training checkpoints are local trusted artifacts.

Measured cold-start adjustment: uniform task placement produced >99% deadline
failures and nearly constant rewards. Initialize a learnable own-ground logit bias
at 4 (all legal candidates keep nonzero probability); optimize it with the actor.
No environment, action-space or reward change. Initial deterministic all-ground
performance must be reported as initialization, not claimed as learned improvement.
Checkpoint format version 2 distinguishes this architecture from exploratory runs.
