"""Paired evaluation of independent distributed SCA versus saved CEM targets.

Run: python -B -m methods.distributed_sca.run --output <directory>
There is no optimizer training, policy update, model fitting or test-based model
selection. The saved CEM target array is passed only to the reference rollout.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import tomllib
import traceback

import numpy as np
import torch

from .engine import Simulator, build_problem
from .planner import coordinate

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    with Path(path).open("rb") as file:
        sections = tomllib.load(file)
    return {**sections["experiment"], **sections["optimizer"]}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_inputs(config):
    checkpoint = torch.load(ROOT / config["source_checkpoint"], map_location="cpu", weights_only=False)
    source = dict(environment=checkpoint["environment"], member_policies={
        key: value for key, value in checkpoint["member_policies"].items() if ".actor." in key})
    reference = torch.load(ROOT / config["cem_checkpoint"], map_location="cpu", weights_only=False)
    source["low_actor_identical"] = all(torch.equal(value, reference["member_policies"][key])
                                        for key, value in source["member_policies"].items())
    source["reference_environment_identical"] = source["environment"] == reference["environment"]
    targets = reference["targets"].numpy().astype(float)
    return source, targets


def run_episode(source, config, epoch, targets, progress_path=None):
    torch.set_num_threads(1)
    sim = Simulator(source, config["slots"])
    env = sim.env
    env.reset(epoch_index=epoch)
    traffic = [None] * env.masters.num_uavs
    neighbors = [tuple(region - 1 for region in env.masters.neighbor_region_ids[master + 1])
                 for master in range(env.masters.num_uavs)]
    count = env.members.member_uav_count
    step = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s
    side = env.region.config.side_length
    slot_delays, planning_times = [], []
    failed, violations, max_increase = 0, 0, 0.
    counters = dict(block_attempts=0, accepted_steps=0, oracle_evaluations=0,
                    neighbor_oracle_calls=0, backtracking_rejections=0)
    started = perf_counter()
    for slot in range(config["slots"]):
        planning_start = perf_counter()
        if targets is None:
            problems = [build_problem(env, master, traffic[master], config["prior_weight"], config["smoothing_m"])
                        for master in range(env.masters.num_uavs)]
            actions, diagnostics = coordinate(problems, neighbors, step, side,
                                               config["rounds"], config["backtracks"])
            max_increase = max(max_increase, diagnostics["proxy_after_s"] - diagnostics["proxy_before_s"])
            for key in counters:
                counters[key] += diagnostics[key]
        else:
            actions = np.clip((targets * side - env.members.positions[:count, :2]) / step, -1., 1.)
        planning_times.append(perf_counter() - planning_start)
        outcome, observed = sim.step(actions)
        if targets is None:
            traffic = observed
        result = outcome.slot_result
        slot_delays.append(sum(dag.completion_time - result.slot_start for dag in result.dags) / len(result.dags))
        failed += sum(dag.failed for dag in result.dags)
        violations += int(env.flight_violations.sum())
        if progress_path is not None and ((slot + 1) % 25 == 0 or slot + 1 == config["slots"]):
            write_json(progress_path, dict(epoch=epoch, method="sca" if targets is None else "cem",
                       slots_completed=slot + 1, total_slots=config["slots"],
                       mean_delay_s=float(np.mean(slot_delays)), elapsed_s=perf_counter() - started,
                       updated_utc=datetime.now(timezone.utc).isoformat()))
    return dict(epoch=epoch, slots_completed=config["slots"], dag_count=config["slots"] * env.users.num_users,
                mean_delay_s=float(np.mean(slot_delays)), failed_dags=int(failed), boundary_violations=violations,
                planning_mean_ms=float(np.mean(planning_times)) * 1000.,
                planning_p95_ms=float(np.quantile(planning_times, .95)) * 1000.,
                elapsed_s=perf_counter() - started, max_proxy_increase_s=max_increase, **counters,
                slot_delay_s=slot_delays, final_positions_m=env.members.positions[:count].tolist())


def compare(records, seed):
    sca = np.array([record["sca"]["mean_delay_s"] for record in records])
    cem = np.array([record["cem"]["mean_delay_s"] for record in records])
    difference = sca - cem
    margin = sca - 1.01 * cem
    indices = np.random.default_rng(seed).integers(len(records), size=(10000, len(records)))
    interval = np.quantile(difference[indices].mean(axis=1), [.025, .975])
    margin_interval = np.quantile(margin[indices].mean(axis=1), [.025, .975])
    return dict(sca_mean_delay_s=float(sca.mean()), cem_mean_delay_s=float(cem.mean()),
                difference_s=float(difference.mean()), relative_difference_pct=float((sca.mean() / cem.mean() - 1.) * 100.),
                paired_difference_ci95_s=interval.tolist(), paired_1pct_margin_ci95_s=margin_interval.tolist(),
                within_1pct=bool(sca.mean() <= 1.01 * cem.mean()), ci_supports_1pct=bool(margin_interval[1] <= 0.))


def _initialize(source, config, targets, output):
    global _SOURCE, _CONFIG, _TARGETS, _OUTPUT
    _SOURCE, _CONFIG, _TARGETS, _OUTPUT = source, config, targets, Path(output)
    torch.set_num_threads(1)


def _pair(epoch):
    sca = run_episode(_SOURCE, _CONFIG, epoch, None, _OUTPUT / f"progress_sca_{epoch}.json")
    cem = run_episode(_SOURCE, _CONFIG, epoch, _TARGETS, _OUTPUT / f"progress_cem_{epoch}.json")
    return dict(epoch=epoch, sca=sca, cem=cem)


def evaluate(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    write_json(output / "config.json", config)
    source, targets = load_inputs(config)
    write_json(output / "metadata.json", dict(
        method="distributed_sca", high_level_neural_network=False, training=False,
        information="Own users and executed traffic, one-hop UAV positions; scalar/partial-gradient neighbor oracles.",
        optimization="3-round block proximal SCA by default; public sequential Master-ID coordination; bounded backtracking.",
        objective="Frozen-traffic weighted link-service delay with an analytical expected-input prior; not exact DAG latency.",
        approximation="Fixed past routing/contention; fixed 1/2 relay prior; 0.1 m smoothed horizontal link norm by default.",
        guarantee="Accepted steps do not increase the current analytic proxy; no finite-round or true-delay optimality guarantee.",
        timing="Planning includes local frame construction and SCA, excludes physical message latency and feedback extraction. CEM timing excludes offline search.",
        interval="Paired bootstrap across independent workload epochs, not across individual DAGs.",
        environment=source["environment"], source_sha256=hashlib.sha256((ROOT / config["source_checkpoint"]).read_bytes()).hexdigest(),
        cem_sha256=hashlib.sha256((ROOT / config["cem_checkpoint"]).read_bytes()).hexdigest(),
        low_actor_identical=source["low_actor_identical"], reference_environment_identical=source["reference_environment_identical"]))
    records = []
    write_json(output / "progress.json", dict(status="running", pairs_completed=0, total_pairs=len(config["epochs"])))

    def record_done(record):
        records.append(record)
        write_json(output / f"episode_{record['epoch']}.json", record)
        write_json(output / "progress.json", dict(status="running", pairs_completed=len(records),
                   total_pairs=len(config["epochs"]), elapsed_s=perf_counter() - started))
        print(json.dumps(dict(epoch=record["epoch"], sca_s=record["sca"]["mean_delay_s"],
                              cem_s=record["cem"]["mean_delay_s"], planning_ms=record["sca"]["planning_mean_ms"])), flush=True)

    if config["workers"] == 1:
        _initialize(source, config, targets, output)
        for epoch in config["epochs"]:
            record_done(_pair(epoch))
    else:
        with ProcessPoolExecutor(max_workers=config["workers"], initializer=_initialize,
                                 initargs=(source, config, targets, str(output))) as pool:
            futures = [pool.submit(_pair, epoch) for epoch in config["epochs"]]
            for future in as_completed(futures):
                record_done(future.result())
    records.sort(key=lambda record: record["epoch"])
    comparison = compare(records, config["seed"])
    valid = all(record[method][metric] == 0 for record in records for method in ("sca", "cem")
                for metric in ("failed_dags", "boundary_violations"))
    report = dict(method="distributed_sca", episodes=records, comparison=comparison,
                  low_actor_identical=source["low_actor_identical"],
                  reference_environment_identical=source["reference_environment_identical"],
                  matches_cem=bool(valid and source["low_actor_identical"] and source["reference_environment_identical"]
                                   and comparison["within_1pct"] and comparison["ci_supports_1pct"]),
                  elapsed_s=perf_counter() - started)
    write_json(output / "results.json", report)
    write_json(output / "progress.json", dict(status="completed", pairs_completed=len(records),
               total_pairs=len(records), elapsed_s=report["elapsed_s"], comparison=comparison))
    print(json.dumps(comparison), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/methods/distributed_sca.toml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--epochs", type=int, nargs="+")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("slots", "epochs", "workers"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    output = args.output or ROOT / config["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    try:
        evaluate(config, output)
    except Exception:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failure.json", dict(error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
