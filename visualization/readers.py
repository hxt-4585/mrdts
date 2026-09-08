"""Validated, concurrent-writer-safe readers for training artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = {"member", "master"}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class EpochRow:
    epoch: int
    stage: str
    stage_epoch: int
    steps: int
    total_steps: int
    epoch_return: float
    mean_reward: float
    mean_delay_s: float


@dataclass(frozen=True)
class UpdateRow:
    epoch: int
    stage: str
    update: int
    total_steps: int
    actor_loss: float
    value_loss: float
    approx_kl: float
    entropy: float
    early_stop: bool


@dataclass(frozen=True)
class ValidationRow:
    epoch: int
    stage: str
    steps: int
    epoch_return: float
    mean_reward: float
    mean_delay_s: float


@dataclass(frozen=True)
class RunData:
    run: Path
    seed: int | None
    mode: str | None
    status: str | None
    epochs: list[EpochRow]
    updates: list[UpdateRow]
    validation: list[ValidationRow]


def _complete_dict_rows(path: Path) -> list[dict[str, str]]:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    newline = content.rfind("\n")
    if newline < 0:
        return []
    complete = content[: newline + 1]
    try:
        return list(csv.DictReader(io.StringIO(complete), strict=True))
    except csv.Error as error:
        raise ValueError(f"Malformed complete CSV data in {path}: {error}") from error


def _required(row: dict[str, str], names: set[str], path: Path, line: int) -> None:
    missing = sorted(name for name in names if row.get(name) in (None, ""))
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} in {path} row {line}")


def _finite(values: list[float], path: Path, line: int) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Values must be finite in {path} row {line}")


def _parse_epoch(row: dict[str, str], path: Path, line: int, validation: bool) -> EpochRow:
    names = {item.name for item in fields(EpochRow)}
    _required(row, names, path, line)
    try:
        parsed = EpochRow(
            epoch=int(row["epoch"]), stage=row["stage"], stage_epoch=int(row["stage_epoch"]),
            steps=int(row["steps"]), total_steps=int(row["total_steps"]),
            epoch_return=float(row["epoch_return"]), mean_reward=float(row["mean_reward"]),
            mean_delay_s=float(row["mean_delay_s"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value in {path} row {line}") from error
    minimum_epoch = 0 if validation else 1
    if (parsed.epoch < minimum_epoch or parsed.stage_epoch < 0 or parsed.steps < 1
            or parsed.total_steps < 0 or parsed.stage not in STAGES):
        raise ValueError(f"Invalid epoch metadata in {path} row {line}")
    _finite([parsed.epoch_return, parsed.mean_reward, parsed.mean_delay_s], path, line)
    if not math.isclose(parsed.epoch_return, parsed.mean_reward * parsed.steps,
                        rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError(f"Reward total is inconsistent in {path} row {line}")
    return parsed


def _read_epoch_rows(path: str | Path) -> list[EpochRow]:
    path = Path(path)
    rows = [_parse_epoch(row, path, line, False)
            for line, row in enumerate(_complete_dict_rows(path), start=2)]
    for previous, current in zip(rows, rows[1:]):
        if current.epoch <= previous.epoch:
            raise ValueError(f"Epoch numbers must be strictly increasing in {path}")
    return rows


def read_epochs(path: str | Path) -> list[EpochRow]:
    return _read_epoch_rows(path)


def read_validation(path: str | Path) -> list[ValidationRow]:
    path = Path(path)
    names = {item.name for item in fields(ValidationRow)}
    rows = []
    for line, row in enumerate(_complete_dict_rows(path), start=2):
        _required(row, names, path, line)
        try:
            item = ValidationRow(
                epoch=int(row["epoch"]), stage=row["stage"], steps=int(row["steps"]),
                epoch_return=float(row["epoch_return"]), mean_reward=float(row["mean_reward"]),
                mean_delay_s=float(row["mean_delay_s"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid numeric value in {path} row {line}") from error
        if item.epoch < 0 or item.steps < 1 or item.stage not in STAGES:
            raise ValueError(f"Invalid validation metadata in {path} row {line}")
        _finite([item.epoch_return, item.mean_reward, item.mean_delay_s], path, line)
        if not math.isclose(item.epoch_return, item.mean_reward * item.steps,
                            rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError(f"Reward total is inconsistent in {path} row {line}")
        rows.append(item)
    rank = {"member": 0, "master": 1}
    for previous, current in zip(rows, rows[1:]):
        if (current.epoch, rank[current.stage]) <= (previous.epoch, rank[previous.stage]):
            raise ValueError(f"Validation keys must be chronological in {path}")
    return rows


def _parse_bool(value: str, path: Path, line: int) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "1.0", "0.0"}:
        raise ValueError(f"Invalid early_stop in {path} row {line}")
    return normalized in {"true", "1", "1.0"}


def read_updates(path: str | Path) -> list[UpdateRow]:
    path = Path(path)
    names = {item.name for item in fields(UpdateRow)}
    parsed = []
    for line, row in enumerate(_complete_dict_rows(path), start=2):
        _required(row, names, path, line)
        try:
            item = UpdateRow(
                epoch=int(row["epoch"]), stage=row["stage"], update=int(row["update"]),
                total_steps=int(row["total_steps"]), actor_loss=float(row["actor_loss"]),
                value_loss=float(row["value_loss"]), approx_kl=float(row["approx_kl"]),
                entropy=float(row["entropy"]), early_stop=_parse_bool(row["early_stop"], path, line),
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, ValueError) and "early_stop" in str(error):
                raise
            raise ValueError(f"Invalid numeric value in {path} row {line}") from error
        if item.epoch < 1 or item.update < 1 or item.total_steps < 0 or item.stage not in STAGES:
            raise ValueError(f"Invalid update metadata in {path} row {line}")
        _finite([item.actor_loss, item.value_loss, item.approx_kl, item.entropy], path, line)
        parsed.append(item)
    for previous, current in zip(parsed, parsed[1:]):
        if (current.epoch, current.update) <= (previous.epoch, previous.update):
            raise ValueError(f"Update keys must be strictly increasing in {path}")
        if current.total_steps < previous.total_steps:
            raise ValueError(f"Update total_steps must be increasing in {path}")
    return parsed


def _json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_run(run: str | Path, *, allow_pending: bool = False) -> RunData:
    run = project_path(run)
    config = _json_object(run / "config.json")
    metadata = _json_object(run / "metadata.json")
    seed = config.get("experiment", {}).get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("config.json experiment.seed must be an integer")
    data = RunData(
        run=run, seed=seed, mode=metadata.get("mode"), status=metadata.get("status"),
        epochs=read_epochs(run / "training" / "epochs.csv"),
        updates=read_updates(run / "training" / "updates.csv"),
        validation=read_validation(run / "training" / "validation.csv"),
    )
    if not allow_pending and not data.epochs:
        raise ValueError(f"No completed epochs found in {run / 'training' / 'epochs.csv'}")
    return data
