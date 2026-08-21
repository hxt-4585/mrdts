# Region and DAG Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Externalize Region and DAG settings, give each generator an independent seed, and make square-region generation smooth and validated.

**Architecture:** TOML files provide default experiment values. Typed settings objects load one file each. Region generation consumes only the region settings and DAG generation owns instance-local Python and NumPy RNGs from DAG settings.

**Tech Stack:** Python standard library `tomllib`, NumPy, unittest.

## Global Constraints

- The global simulation area is square and `grid_size` is the number of cells per side.
- Region and DAG random streams are independent.
- No third-party configuration dependency is added.

---

### Task 1: Add external typed settings

**Files:**
- Create: `config/region.toml`
- Create: `config/dag.toml`
- Create: `env/settings/__init__.py`
- Create: `env/settings/region.py`
- Create: `env/settings/dag.py`
- Test: `tests/test_settings.py`

- [ ] Write tests that load the repository TOML files and assert typed defaults, derived cell size, and distinct region/DAG seeds.
- [ ] Run the tests and observe import/load failures.
- [ ] Implement small typed loaders using `tomllib`; derive cell size from `side_length / grid_size`.
- [ ] Re-run settings tests.

### Task 2: Refactor generators to the settings contracts

**Files:**
- Modify: `env/graph_utils.py`
- Modify: `env/region.py`
- Modify: `scripts/visualize_dag.py`
- Modify: `scripts/visualize_region.py`
- Modify: `tests/test_dag_generation.py`
- Modify: `tests/test_region.py`
- Delete: `env/config.py`

- [ ] Write tests proving DAG output is reproduced by the same DAG seed despite changes to global random state, and region coordinates use square-grid indexing.
- [ ] Run those tests and observe the old global-RNG/import behaviour.
- [ ] Give `DAGGenerator` local RNG instances; migrate all callers to `env.settings`; replace independent grid-axis configuration with `grid_size`.
- [ ] Re-run unit tests.

### Task 3: Smooth region expansion and map persistence checks

**Files:**
- Modify: `env/region.py`
- Modify: `tests/test_region.py`

- [ ] Write tests for invalid partition inputs and loading a map with mismatched shape.
- [ ] Run the new tests and observe failures.
- [ ] Validate region settings before generation and map shape on load; favour boundary candidates with more same-region neighbours according to `shape_smoothness`.
- [ ] Run all tests and visualization scripts.
