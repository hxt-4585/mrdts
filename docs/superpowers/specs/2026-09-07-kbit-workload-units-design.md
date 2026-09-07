# Kbit workload units

## Goal

Interpret DAG node input sizes and dependency-edge result sizes as decimal kilobits throughout the simulator: `1 Kbit = 1000 bit`.

## Design

- Keep the existing TOML field names and numeric ranges unchanged.
- Change both runtime transmission-size conversions from the old `KB * 1024 * 8` rule to `Kbit * 1000`.
- Keep ERS estimates and actual event execution on the same conversion rule.
- Update configuration comments, code documentation, specifications, and tests to name Kbit explicitly.
- Do not add a configurable unit or compatibility mode; all current workload values use Kbit.

## Verification

- A node input value of `1.0` must become exactly `1000` bits in `TaskRuntime`.
- An edge result value of `1.0` must produce the same duration as a direct `1000`-bit runtime transfer.
- The focused unit tests and the complete test suite must pass.
