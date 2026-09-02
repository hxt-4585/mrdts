# Ground-Device Local Execution Design

## Goal

Allow the Member UAV scheduling decision for each DAG subtask to choose exactly one of:

1. the DAG's own ground device;
2. a Member UAV in the owner's current region;
3. the global BS.

A subtask must never execute on another ground device. Ground-device computation contributes delay and occupies its local CPU, but its computation energy is ignored.

## Compute resources

Each ground device owns exactly one independent CPU core. The core count is an invariant rather than a configurable degree of freedom. The default core frequency is `1e9` CPU cycles/s and remains configurable through `config/user.toml`. When a scheduling runtime is created, every ground device is registered as its own `ServerState` with `capacitance_factor=0.0`. `SchedulingRuntime` rejects missing, multi-core, or non-zero-capacitance ground servers, so direct construction cannot bypass this rule. This reuses the existing FIFO compute lifecycle and makes `TaskRuntime.compute_energy_j` naturally equal to zero for local execution.

## Placement validation

`PlacementDecision.execution_node` continues to use `EntityRef`. Submission validation accepts:

- the submitting DAG's own `ground_device`;
- any registered Member UAV in `owner_member`'s current region;
- the global BS.

Any other ground device and any unknown entity are rejected.

Submission is fail-closed: the current Member-region map and complete Ground-owner association map must be installed before a DAG is accepted. The runtime records exactly one global BS and rejects every other BS identity.

## Routing

Raw input for a locally executed subtask has an empty route and is considered available at `epoch_start`. Raw input for an offloaded subtask continues to pass through the associated Member UAV.

Intermediate results use the associated Member UAV as the access relay whenever one endpoint is the ground device:

| Source | Target | Route |
| --- | --- | --- |
| same entity | same entity | local, no transfer |
| ground | owner Member | ground -> owner |
| ground | other Member or BS | ground -> owner -> target |
| owner Member | ground | owner -> ground |
| other Member or BS | ground | source -> owner -> ground |
| Member or BS | Member or BS | source -> target |

This avoids introducing a direct ground-to-BS channel and preserves the Member UAV's role as the ground device's access node.

## Energy accounting

Ground-device computation energy is always zero. Existing transfer records may still record transmitter energy because transmit power is required to calculate the link rate; a future reward implementation must exclude ground-device energy if the objective is strictly UAV energy.

## ERS integration

Local tasks receive ordinary `ers_seq` values and enter the ground device's server queue under the same ordering rules as UAV and BS tasks. The later unified multi-DAG ERS implementation must include the local CPU as a candidate when calculating average computation costs.

## Verification

Tests must cover:

- user CPU-frequency configuration and one-core server creation;
- empty raw-input route for local execution;
- ground/edge mixed predecessor routes through the owner Member;
- an all-local DAG finishing with no channel use and zero computation energy;
- rejection of placement on another ground device;
- existing Member/BS scheduling behavior remaining unchanged.
