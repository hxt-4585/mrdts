"""时隙调度窗口配置。"""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class SchedulingConfig:
    max_duration_s: float

    def __post_init__(self):
        if not math.isfinite(self.max_duration_s) or self.max_duration_s <= 0.0:
            raise ValueError("max_duration_s 必须为有限正数")

    @classmethod
    def default(cls) -> "SchedulingConfig":
        return cls.from_toml(Path(__file__).resolve().parents[2] / "config" / "scheduling.toml")

    @classmethod
    def from_toml(cls, path: str | Path) -> "SchedulingConfig":
        with Path(path).open("rb") as file:
            return cls(**tomllib.load(file)["scheduling"])
