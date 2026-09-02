# Ground-Device Local Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support source-ground-device execution as a first-class placement with one CPU core, scheduling delay, and zero computation energy.

**Architecture:** Register each ground device as an ordinary zero-capacitance `ServerState`, extend routing around the owner Member access relay, and handle zero-hop input readiness in the event runtime. Keep `PlacementDecision` unchanged and enforce the allowed placement set at submission validation.

**Tech Stack:** Python 3.11, NumPy, `unittest`, TOML configuration.

## Global Constraints

- A ground device has exactly one CPU core.
- Default ground CPU frequency is `1e9` CPU cycles/s.
- A task may execute only on its own ground device, a same-region Member UAV, or the global BS.
- Ground-device computation energy is zero.
- Existing user changes and unrelated files remain untouched.

---

### Task 1: Ground compute configuration and server registration

**Files:**
- Modify: `config/user.toml`
- Modify: `env/settings/user_config.py`
- Modify: `env/environment.py`
- Test: `tests/test_settings.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Produces: `UserConfig.core_frequency: float`
- Produces: one `ServerState` per ground device with capacitance factor `0.0`

- [x] Add failing tests asserting the default user core count/frequency and runtime ground servers.
- [x] Run `uv run python -B -m unittest tests.test_settings tests.test_environment -v` and confirm failures reference missing ground compute configuration/server registration.
- [x] Add `core_frequency = 1000000000.0` to user configuration, load it through `UserConfig`, and register each ground server with one core and zero capacitance.
- [x] Re-run the focused tests and confirm they pass.

### Task 2: Ground-aware routing and placement validation

**Files:**
- Modify: `env/routing.py`
- Modify: `env/event_runtime.py`
- Test: `tests/test_routing.py`
- Test: `tests/test_event_runtime.py`

**Interfaces:**
- Produces: `RoutePlanner.predecessor_route(source_node, target_node, ground_device, owner_member) -> RoutePlan`
- Produces: placement validation that accepts only the source ground, same-region Members, or BS

- [x] Add failing routing tests for local input, ground-to-edge, and edge-to-ground relay paths.
- [x] Add a failing runtime test rejecting a different ground device as execution node.
- [x] Run `uv run python -B -m unittest tests.test_routing tests.test_event_runtime -v` and confirm the expected failures.
- [x] Implement the route rules and strict placement validation.
- [x] Re-run the focused tests and confirm they pass.

### Task 3: Zero-hop input readiness and end-to-end local execution

**Files:**
- Modify: `env/event_runtime.py`
- Test: `tests/test_event_runtime.py`
- Modify: `specs/01_system_model.md`
- Modify: `handoff.md`

**Interfaces:**
- Produces: local raw input available at `epoch_start` without a transfer event
- Produces: all-local DAG execution through the ordinary server lifecycle with zero compute energy

- [x] Add a failing all-local DAG test asserting no channel is created, both tasks finish, and computation energy is zero.
- [x] Run `uv run python -B -m unittest tests.test_event_runtime -v` and confirm the task remains waiting before implementation.
- [x] Mark empty-route input as immediately arrived and enqueue it when all dependencies are ready.
- [x] Update the system model and handoff with local execution semantics and the exact action candidate set.
- [x] Run `uv run python -B -m unittest discover -s tests -v` and confirm the complete suite passes.
- [x] Run `git diff --check` and inspect `git diff` for scope and accidental changes.
