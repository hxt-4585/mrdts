"""时隙调度窗口配置。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class SchedulingConfig:
    max_duration_s: float

    @classmethod
    def default(cls) -> "SchedulingConfig":
        return cls.from_toml(Path(__file__).resolve().parents[2] / "config" / "scheduling.toml")

    @classmethod
    def from_toml(cls, path: str | Path) -> "SchedulingConfig":
        with Path(path).open("rb") as file:
            return cls(**tomllib.load(file)["scheduling"])
