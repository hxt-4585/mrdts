# Region and DAG Configuration Design

## Goal

Move experiment parameters out of Python source files, make region and DAG random streams independent, and simplify the region geometry contract.

## Configuration

Repository TOML files are the source of default values:

- `config/region.toml`: square-map geometry and partition parameters.
- `config/dag.toml`: DAG topology, feature ranges, and its independent seed.

The region map is a square with `side_length` metres per side and `grid_size` cells per side. `cell_size` is derived. Region configuration has `region_count`, `min_area_ratio`, `area_imbalance`, `shape_smoothness`, and `seed`.

## Module boundaries

- `env/settings/region.py` and `env/settings/dag.py` parse their respective TOML file into typed configuration objects.
- `env/region.py` generates and queries a region map. Its expansion policy favours candidates that share more edges with the same region, controlled by `shape_smoothness`, to discourage single-cell spurs.
- `env/graph_utils.py` owns two instance-local RNGs seeded by the DAG configuration. It never reads or mutates global `random` or `numpy.random` state.

## Compatibility and error behaviour

The old `env.config` module is removed after all project callers migrate to `env.settings`. Region configuration rejects impossible geometry or partition input before generation. A stored region map must match the configured square grid before it is accepted.

## Verification

Tests load the checked-in TOML defaults, assert independent/reproducible RNG behaviour, exercise square coordinate lookup, preserve DAG acyclicity, and verify region coverage, connectivity, and area limits.
