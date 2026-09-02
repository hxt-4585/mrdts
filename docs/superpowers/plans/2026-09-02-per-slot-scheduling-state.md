# Per-Slot Scheduling State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the duplicate UAV-side core availability state and align project documentation with a fresh scheduling runtime per slot.

**Architecture:** `MemberUAV` retains only static compute capacity. `SchedulingRuntime` owns the current slot's channel, server, core, and DAG state through `ServerState`; the runtime is discarded at the slot boundary.

**Tech Stack:** Python 3.11, NumPy, `unittest`, Markdown.

## Global Constraints

- Each slot creates a fresh `SchedulingRuntime`.
- Scheduling state and unfinished DAGs do not cross slot boundaries.
- Fixed 0.5-second flight time is outside the one-second scheduling event timeline.
- Do not implement global multi-DAG ERS in this change.

---

### Task 1: Specify the removed public state and APIs

**Files:**
- Modify: `tests/test_uav.py`
- Modify: `tests/test_environment.py`

**Interfaces:**
- Consumes: generated `MemberUAV` and `Environment` instances.
- Produces: regression tests requiring the obsolete attributes and methods to be absent.

- [ ] **Step 1: Write failing removal tests**

```python
def test_does_not_store_runtime_core_availability(self):
    members = self._generate_members()
    self.assertFalse(hasattr(members, "core_available_at"))
    self.assertFalse(hasattr(members, "schedule_computation"))
```

Add an environment test asserting that `estimate_finish_time` and `reserve_computation` are absent.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run python -B -m unittest tests.test_uav tests.test_environment -v`

Expected: FAIL because the four obsolete names still exist.

### Task 2: Remove duplicate compute state

**Files:**
- Modify: `env/uav.py`
- Modify: `env/environment.py`
- Modify: `tests/test_uav.py`
- Modify: `tests/test_environment.py`

**Interfaces:**
- Consumes: `MemberUAV.core_counts`, `MemberUAV.core_frequencies`, and `SchedulingRuntime.servers`.
- Produces: UAV entities with static compute capacity only and an environment without legacy reservation APIs.

- [ ] **Step 1: Delete UAV-side availability initialization and scheduling**

Remove `self.core_available_at`, its zero initialization, and `schedule_computation()` from `MemberUAV`.

- [ ] **Step 2: Delete environment wrappers**

Remove `estimate_finish_time()` and `reserve_computation()` from `Environment`, and update its class description to name `SchedulingRuntime` as the owner of per-slot compute state.

- [ ] **Step 3: Remove obsolete behavioral tests**

Delete tests that exercise the removed scheduling and reservation APIs, retaining the new API-removal regression tests.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `uv run python -B -m unittest tests.test_uav tests.test_environment -v`

Expected: PASS.

### Task 3: Align lifecycle documentation

**Files:**
- Modify: `handoff.md`
- Modify: `env/environment.py`
- Modify: `env/event_runtime.py`

**Interfaces:**
- Consumes: the approved per-slot lifecycle design.
- Produces: documentation that consistently describes a fresh runtime, failed unfinished DAGs, and flight outside the scheduling clock.

- [ ] **Step 1: Replace contradictory handoff statements**

State that runtime queues are per-slot, unfinished DAGs fail at the deadline, and the fixed flight duration is not represented in the scheduling event clock. Remove references to updating `core_available_at`.

- [ ] **Step 2: Update code documentation**

Change `create_scheduling_runtime()` and topology-refresh docstrings so they no longer promise cross-slot persistence.

- [ ] **Step 3: Scan for contradictions**

Run:

```powershell
rg -n "core_available_at|跨时隙保留|可跨时隙保存|未完成 DAG 必须跨时隙|绝对时间线" handoff.md specs docs env methods tests
```

Expected: no stale lifecycle claims outside historical text that is explicitly marked obsolete.

### Task 4: Full verification and review

**Files:**
- Review all modified files.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a verified, reviewable change set.

- [ ] **Step 1: Run the complete test suite**

Run: `uv run python -B -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 2: Inspect the final diff and repository state**

Run: `git diff --check`, `git diff --stat`, and `git status --short`.

Expected: no whitespace errors and only scoped files changed.

- [ ] **Step 3: Report exact documentation changes**

List every corrected lifecycle statement and distinguish retained physical cross-slot state from removed scheduling cross-slot state.
