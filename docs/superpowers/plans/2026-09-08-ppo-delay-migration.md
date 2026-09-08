# PPO Delay Migration Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for isolated tasks and reviews. Steps below track implementation.

**Goal:** Port the recovered working PPO baseline into the modular project with reusable learning code, standard experiments and training plots.

**Architecture:** Preserve source learning semantics. Replace the legacy scene/CLI/output plumbing with the current Scene and experiment contracts. Keep generic PPO independent of UAV details.

**Tech Stack:** Python 3.11, project-locked PyTorch, NumPy, Matplotlib, unittest.

## Global Constraints

- Source is c47059dee760e0095b7d135a856a114fa63e1a5e.
- Reward only negative normalized global censored DAG delay; current Kbit units.
- Result layout matches Random; visualization/ is a sibling of results/.
- Default 500 slots, Member update every 8; continuous Master, shared Member, ground bias 4, centralized critics.
- No changes to physical environment semantics; no publishing or merging.

## Task 1: Shared learning and method execution

Files: methods/learning/{networks/mlp,buffers/rollout,algorithms/ppo}.py;
methods/solutions/ppo/{networks,observations,reward,method,rollout}.py;
methods/components/{flight,scheduling}/ppo.py; tests/test_ppo.py.

- [x] Add tests for masked probabilities, Kbit features, legal complete plans, true terminals and staged updates; observe missing implementation failure.
- [x] Extract source network/PPO code using `git show <source>:methods/rl_baseline/<file>`; retain formulas and settings, split generic types/helpers, add explicit tensor devices.
- [x] Implement `PPOMethod.run_slot(scene, workload)` returning SlotOutcome and `Learner` + `run_epoch(scene, learner, stage, steps, update_every, on_step, on_update)` using the same method path.
- [x] Verify with `.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_ppo.py` and compare source/new network outputs and PPO updates on identical synthetic samples.

## Task 2: Training, checkpoint and experiments

Files: solution {trainer,checkpoint,settings,logs}.py; experiments/{cli,config,runner,train,aggregate}.py; methods/factory.py; config/{experiments,methods}/ppo.toml; tests/test_ppo_experiments.py.

- [x] Test unique run layout, checkpoint evaluation, preserved Random behavior, split IDs and exact epoch-boundary resume.
- [x] Implement Trainer.train(config, device), deterministic method loading, source-state schema validation, epoch/update/validation logging, latest/best selection and interruption reconciliation.
- [x] Keep metrics.csv common slot schema; add learning CSVs under training/. Record actual episode IDs, steps and checkpoint provenance.
- [x] Run short staged CLI training/evaluation/resume; compare uninterrupted and resumed parameters and metrics; run all existing experiments tests.

## Task 3: Visualization (independent)

Files: visualization/{__init__,readers,style,training,watch}.py; tests/test_training_visualization.py; visualization/README.md.

- [x] Test readers with real fixture logs including partial last lines, stages, empty data and inconsistent reward totals.
- [x] Implement `python -m visualization.training --run RUN` and `python -m visualization.watch --run RUN [--once]`. Read training/epochs.csv and updates.csv; optional validation.csv; save figures/training/.
- [x] Render reward, actor/value loss and diagnostics; retain raw points and within-stage smoothing, separate validation; export source CSV with offline figures.
- [x] Verify exported image and fixture tests. Use no training imports or GUI dependency in offline mode.

## Task 4: Integrated validation and documentation

- [x] Review all changes for interface/semantic parity, fix findings, run the complete unittest suite.
- [x] Run bounded real training on the default 100-user workload, evaluate paired initial/final models, record actual results without claiming general convergence.
- [x] Update README, PROJECT_STRUCTURE and solution usage; record validation evidence and final commands.
- [x] Commit final documentation and long-epoch verification on codex/development; leave main/remote unchanged.

Validation details and actual measurements: `docs/research/2026-09-08-ppo-delay-migration-verification.md`. Shared core, experiment integration and visualization have been reviewed, tested and committed separately. The 500-slot Member and Master check completed with 63 + 1 finite updates; final verification and usage documentation are included with this plan.
