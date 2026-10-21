"""飞行、调度窗口、epoch 步数与随机种子的环境配置。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class EnvConfig:
    flight_duration_s: float
    scheduling_duration_s: float
    steps_per_epoch: int
    env_seed: int

    @classmethod
    def default(cls) -> "EnvConfig":
        return cls.from_toml(Path(__file__).resolve().parents[2] / "config" / "env.toml")

    @classmethod
    def from_toml(cls, path: str | Path) -> "EnvConfig":
        with Path(path).open("rb") as file:
            return cls(**tomllib.load(file)["env"])
