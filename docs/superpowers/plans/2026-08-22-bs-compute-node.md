# BS Compute Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed BS as the final vectorized compute-node row alongside Member UAVs.

**Architecture:** `MemberUAV` owns a single `(M + 1)` numerical view: Member rows come first and the BS is always the final row. Configuration supplies BS placement and resource capacities; Member-only behavior is bounded by `bs_index`.

**Tech Stack:** Python 3.11, NumPy, TOML, unittest, uv.

## Global Constraints

- `bs_index == member_uav_count`; no `is_bs` field is introduced.
- The BS has region ID `0`, fixed center position, 4 cores at `12e9` cycles/s.
- Each Member has 2 cores at `10e9` cycles/s.
- Flight actions have shape `(member_uav_count, 2)` and must not update the BS row.
- Do not implement the communication channel or scheduling policy in this change.

---

### Task 1: Configure and expose the BS compute-node contract

**Files:** `config/uav.toml`, `env/settings/uav_config.py`, `env/uav.py`, `tests/test_uav.py`

- [x] Add failing tests that assert a final BS row at `(500, 500, 25)`, region ID `0`, Member cores `(2, 10e9)`, and BS cores `(4, 12e9)`.
- [x] Run `uv run python -B -m unittest tests.test_uav -v` and confirm the new tests fail.
- [x] Add BS and core settings to `UAVConfig`, append the BS row during `MemberUAV` generation, and expose padded core matrices.
- [x] Run `uv run python -B -m unittest tests.test_uav -v` and confirm the tests pass.

### Task 2: Preserve Member-only movement boundaries

**Files:** `env/uav.py`, `tests/test_uav.py`

- [x] Add a failing test that a valid `(member_uav_count, 2)` action leaves the final BS row unchanged.
- [x] Run the focused test and confirm it fails because actions currently use all rows.
- [x] Update `apply_flight_actions` to validate and update only `[:bs_index]`.
- [x] Run `uv run python -B -m unittest discover -s tests -v` and confirm the full suite passes.
