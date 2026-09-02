# Environment Slot Lifecycle Implementation Plan

**Goal:** Implement the approved begin/end lifecycle with bounded, permanently closed runtimes and failed unfinished DAGs.

**Architecture:** Environment owns one active runtime. SchedulingRuntime owns its bounded event window and immutable results. Physical topology refresh precedes construction. Implement inline in the current workspace; existing approval covers this design.

**Tech Stack:** Python 3.11, NumPy, unittest, TOML, Markdown.

## Constraints

- Preserve ERS ordering, placement candidates and routing.
- Flight remains outside the scheduling clock; use the existing one-second default configuration.
- Preserve completed traces; discard operational state at the deadline, and reject old-runtime reuse.
- Handoff remains ignored. Training reset/step and rewards remain separate work.

## 1. Specify and implement lifecycle behavior

Files: `tests/test_slot_lifecycle.py`, `env/environment.py`, `env/event_runtime.py`, `env/dag_runtime.py`, `env/task_runtime.py`, `env/slot_result.py`, `env/settings/scheduling_config.py`, `env/settings/__init__.py`.

- [x] Add real-entity tests exercising this public sequence:
  ```python
  violations = environment.begin_slot(region, users.positions, actions)
  runtime = environment.runtime
  runtime.submit_dag(..., epoch_start=runtime.slot_start)
  result = environment.end_slot()
  assert runtime.closed and environment.runtime is None
  assert result.failed_dags == (dag_key,)
  ```
- [x] Run `uv run python -B -m unittest discover -s tests -p test_slot_lifecycle.py -v` and observe missing lifecycle failures.
- [x] Implement the documented interfaces. `begin_slot` rejects overlap before flight, refreshes topology without a runtime, constructs the new bounded runtime, and restores physical state if preparation fails. `end_slot` calls `finish_slot`, advances the next start time, and drops its reference.
- [x] Add runtime window validation and permanent closure. Finish events at the deadline, skip dispatch there, mark unfinished work failed, settle partial energy, then clear operational queues and bindings. Return frozen scalar/tuple results with no runtime references.
- [x] Run focused lifecycle tests until all pass.

## 2. Migrate callers and verify

Files: `tests/test_environment.py`, `tests/test_multi_slot_scheduling.py`, `tests/test_event_runtime.py`, `config/scheduling.toml`, `specs/01_system_model.md`, `handoff.md`.

- [x] Refresh topology before factory calls; remove the obsolete runtime argument.
- [x] Replace manual flight/factory orchestration in the ten-slot test with `begin_slot`, and use `end_slot` for finalization and failure assertions.
- [x] Update event tests to advance within their bounded window and assert topology is frozen after submission.
- [x] Run `uv run python -B -m unittest discover -s tests -q` and `git diff --check`.
- [x] Update docs/handoff to distinguish implemented lifecycle from pending RL policy, reward and unified ERS work. Review the final diff and report validation.

Verification: 103 tests passed after lifecycle implementation, including 11 new lifecycle tests. The subsequent scheduling replay adds 3 tests; the full pre-commit suite passes all 106 tests. Diff checks passed. Implementation baseline: `b00757a`.
