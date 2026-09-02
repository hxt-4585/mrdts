"""地面用户的 TOML 配置。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class UserConfig:
    """用户总量与按区域分配的参数。"""

    total_users: int
    min_users_per_region: int
    area_fluctuation: float
    center_bias: float
    center_spread_ratio: float
    seed: int | None
    transmit_power: float
    core_frequency: float

    @classmethod
    def default(cls) -> "UserConfig":
        """加载项目默认用户配置。"""
        return cls.from_toml(Path(__file__).resolve().parents[2] / "config" / "user.toml")

    @classmethod
    def from_toml(cls, path: str | Path) -> "UserConfig":
        """从用户 TOML 文件读取配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)
        population = data["population"]
        computation = data["computation"]
        return cls(
            total_users=population["total_users"],
            min_users_per_region=population["min_users_per_region"],
            area_fluctuation=population["area_fluctuation"],
            center_bias=population["center_bias"],
            center_spread_ratio=population["center_spread_ratio"],
            seed=population.get("seed"),
            transmit_power=population["transmit_power"],
            core_frequency=computation["core_frequency"],
        )
