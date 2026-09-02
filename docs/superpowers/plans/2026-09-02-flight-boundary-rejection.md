# Flight Boundary Rejection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject Member UAV flight actions that would leave the square deployment area and return a boolean violation mask for future RL penalties.

**Architecture:** `MemberUAV.apply_flight_actions` receives the region side length, computes all candidate positions, applies only in-bounds rows, and returns one boolean per Member. Reward calculation remains outside the entity.

**Tech Stack:** Python 3.11, NumPy, `unittest`, Markdown.

## Global Constraints

- Deployment coordinates use the closed interval `[0, side_length]` on both horizontal axes.
- A violating Member keeps both original horizontal coordinates.
- The return value has shape `(member_uav_count,)` and boolean dtype.
- Do not change coverage-radius or ERS behavior.

---

### Task 1: Specify flight boundary behavior

**Files:**
- Modify: `tests/test_uav.py`

**Interfaces:**
- Consumes: `MemberUAV.apply_flight_actions(normalized_actions, side_length)`.
- Produces: tests for legal movement, whole-action rejection, violation shape, and BS immobility.

- [ ] Add a failing test where one Member crosses the lower x boundary while another valid Member moves:

```python
original_positions = members.positions.copy()
members.positions[0, :2] = [1.0, 1.0]
actions = np.zeros((members.member_uav_count, 2), dtype=np.float32)
actions[0] = [-1.0, 0.0]
actions[1] = [0.2, 0.0]
violations = members.apply_flight_actions(actions, region.config.side_length)
self.assertTrue(violations[0])
self.assertFalse(violations[1])
np.testing.assert_array_equal(members.positions[0, :2], [1.0, 1.0])
self.assertGreater(members.positions[1, 0], original_positions[1, 0])
```

- [ ] Update the existing legal-flight test to expect an all-false boolean array returned from `apply_flight_actions`.
- [ ] Run `uv run python -B -m unittest tests.test_uav -v` and verify failure because the method does not accept `side_length` or return violations.

### Task 2: Implement rejection and update callers

**Files:**
- Modify: `env/uav.py`
- Modify: `tests/test_uav.py`
- Modify: `tests/test_multi_slot_scheduling.py`

**Interfaces:**
- Consumes: normalized Member actions and a finite positive `side_length`.
- Produces: mutated valid Member positions and `np.ndarray` boolean violations.

- [ ] Implement validation, candidate calculation, selective update, and the return mask:

```python
side_length = float(side_length)
if not np.isfinite(side_length) or side_length <= 0.0:
    raise ValueError("side_length 必须为有限正数")
candidate_positions = self.positions[: self.bs_index, :2] + (
    normalized_actions * self.config.max_horizontal_speed * self.config.flight_duration
)
in_bounds = ((candidate_positions >= 0.0) & (candidate_positions <= side_length)).all(axis=1)
self.positions[: self.bs_index, :2][in_bounds] = candidate_positions[in_bounds]
return ~in_bounds
```
- [ ] Pass `region.config.side_length` from every caller.
- [ ] Run `uv run python -B -m unittest tests.test_uav tests.test_multi_slot_scheduling -v` and verify success.

### Task 3: Align documentation and verify

**Files:**
- Modify: `specs/01_system_model.md`
- Modify: `handoff.md`

**Interfaces:**
- Consumes: the approved boundary behavior.
- Produces: project documentation describing rejection and future penalty consumption.

- [ ] Document that an out-of-bounds action leaves the Member at its current position and emits a violation flag.
- [ ] Remove the stale handoff statement that boundary handling is unimplemented.
- [ ] Run `uv run python -B -m unittest discover -s tests -v`.
- [ ] Run `git diff --check` and review the scoped diff.
