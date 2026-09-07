# Kbit Workload Units Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Interpret DAG node input sizes and dependency-edge result sizes as decimal Kbit, with `1 Kbit = 1000 bit`, in both actual scheduling and ERS estimates.

**Architecture:** Keep the existing DAG arrays and TOML schema unchanged. Convert workload values at the two consumers—the event runtime and ERS cost model—using the same `1000` multiplier, then align user-facing unit labels and regression tests.

**Tech Stack:** Python 3.11, NumPy, `unittest`, TOML, Markdown

## Global Constraints

- `1 Kbit = 1000 bit`.
- Existing numeric ranges and configuration keys remain unchanged.
- ERS estimates and actual event execution must use the identical conversion.
- Preserve the user's existing `config/experiments/random.toml` modification.

---

### Task 1: Lock down runtime Kbit conversion

**Files:**
- Modify: `tests/test_event_runtime.py`
- Modify: `env/runtime/event_runtime.py:237-275`

**Interfaces:**
- Consumes: `DAG.node_features[:, 0]` and `DAG.edge_features[(parent, child)]`, both expressed in Kbit.
- Produces: `TaskRuntime.input_bits` and transfer payloads expressed in bit.

- [x] **Step 1: Write the failing test**

Update the two-hop runtime test to calculate the expected one-hop duration from exactly `1000` bits and assert that a `1.0` Kbit node input and edge result each use that duration. Also assert `runtime.trace(key).input_bits == 1000.0`.

- [x] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_event_runtime.TestSchedulingRuntime.test_relayed_inputs_and_local_predecessor_enable_fifo_computation -v`

Expected: FAIL because the current runtime stores `8192.0` bits for a `1.0` workload value.

- [x] **Step 3: Write minimal implementation**

In `_prepare_dag`, replace both conversions:

```python
input_bits=float(dag.node_features[node, 0]) * 1000.0
result_bits = float(dag.edge_features[(parent, child)]) * 1000.0
```

- [x] **Step 4: Run focused runtime tests**

Run: `uv run python -m unittest tests.test_event_runtime tests.test_ers_runtime -v`

Expected: runtime Kbit assertions pass; ERS expectations may still fail until Task 2.

### Task 2: Align ERS estimates with Kbit

**Files:**
- Modify: `tests/test_ers.py`
- Modify: `methods/components/ordering/ers.py:142-148`

**Interfaces:**
- Consumes: `DAG.edge_features` in Kbit and `_LocationCosts.seconds_per_bit` in seconds per bit.
- Produces: `AverageCosts.average_edge_comm_s` in seconds.

- [x] **Step 1: Write the failing ERS test**

Rename Kbit-oriented test helpers and make a `1000` Kbit edge represent `1e6` bits. For a `1.0` Kbit edge, assert the average communication formula is multiplied by `1000 / 1e6`.

- [x] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_ers.TestERS.test_asymmetric_routes_and_kbit_to_bits -v`

Expected: FAIL because ERS still multiplies edge sizes by `8192`.

- [x] **Step 3: Write minimal implementation**

Use decimal Kbit in `_task_costs`:

```python
{edge: float(size) * 1000.0 * locations.seconds_per_bit
 for edge, size in dag.edge_features.items()}
```

- [x] **Step 4: Run focused scheduling tests**

Run: `uv run python -m unittest tests.test_ers tests.test_ers_runtime tests.test_event_runtime -v`

Expected: PASS.

### Task 3: Align unit labels and verify the repository

**Files:**
- Modify: `config/dag.toml`
- Modify: `env/workload/dag_generator.py`
- Modify: `specs/03_task_dag_model.md`
- Modify: `docs/superpowers/specs/2026-09-03-ers-delay-design.md`
- Modify: `docs/superpowers/plans/2026-09-03-ers-delay.md`
- Modify: `tests/test_dag_generation.py`
- Modify: `scripts/visualize_dag.py`
- Modify: `scripts/visualize_scheduling.py`
- Modify: `scripts/scheduling_replay.md`
- Modify: `scripts/templates/scheduling_replay.html`

**Interfaces:**
- Consumes and produces no new runtime API; this task aligns labels with the established Kbit contract.

- [x] **Step 1: Replace workload unit labels**

Change `KB` to `Kbit`, rename local display/helper fields such as `input_kb` and `edge_kb` to `input_kbit` and `edge_kbit`, and convert displayed runtime bits with `/ 1000`.

- [x] **Step 2: Scan for stale byte-based semantics**

Run: `rg -n -i "\\bKB\\b|8192|1024\\s*\\*\\s*8|input_kb|edge_kb" . -g '!results/**' -g '!scripts/diagnostics/**' -g '!**/.git/**'`

Expected: no stale workload-unit matches except historical wording explicitly describing the old conversion in the new design document.

- [x] **Step 3: Run the complete test suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected: all tests pass.

- [x] **Step 4: Check the final diff**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; only Kbit-related files plus the user's pre-existing `config/experiments/random.toml` change are present.
