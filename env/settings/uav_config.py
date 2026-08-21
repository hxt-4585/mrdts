"""UAV 的 TOML 配置。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class UAVConfig:
    """两类 UAV 共用的配置入口。"""

    master_uav_count: int
    master_altitude: float
    member_uav_count: int
    min_members_per_region: int
    member_altitude: float
    member_slot_energy: float
    flight_duration: float
    max_horizontal_speed: float
    member_core_count: int
    member_core_frequency: float
    bs_altitude: float
    bs_core_count: int
    bs_core_frequency: float

    @classmethod
    def default(cls) -> "UAVConfig":
        """加载项目默认 UAV 配置。"""
        return cls.from_toml(Path(__file__).resolve().parents[2] / "config" / "uav.toml")

    @classmethod
    def from_toml(cls, path: str | Path) -> "UAVConfig":
        """从 UAV TOML 文件读取配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)
        master = data["master"]
        member = data["member"]
        bs = data["bs"]
        return cls(
            master_uav_count=master["uav_count"],
            master_altitude=master["altitude"],
            member_uav_count=member["uav_count"],
            min_members_per_region=member["min_uavs_per_region"],
            member_altitude=member["altitude"],
            member_slot_energy=member["slot_energy"],
            flight_duration=member["flight_duration"],
            max_horizontal_speed=member["max_horizontal_speed"],
            member_core_count=member["core_count"],
            member_core_frequency=member["core_frequency"],
            bs_altitude=bs["altitude"],
            bs_core_count=bs["core_count"],
            bs_core_frequency=bs["core_frequency"],
        )
