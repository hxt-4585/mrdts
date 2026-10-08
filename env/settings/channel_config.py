"""通信信道的 TOML 配置。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ChannelConfig:
    """三类链路带宽与简化 LoS/NLoS 传播参数。"""

    ground_to_air_bandwidth_mhz: float
    air_to_air_bandwidth_mhz: float
    air_to_ground_bandwidth_mhz: float
    reference_channel_gain: float
    nlos_attenuation_factor: float
    noise_power_w: float
    los_alpha: float
    los_beta: float

    @classmethod
    def default(cls) -> "ChannelConfig":
        """加载项目默认通信配置。"""
        return cls.from_toml(Path(__file__).resolve().parents[2] / "config" / "channel.toml")

    @classmethod
    def from_toml(cls, path: str | Path) -> "ChannelConfig":
        """从通信 TOML 文件读取配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)
        bandwidth = data["bandwidth"]
        propagation = data["propagation"]
        los_probability = propagation["los_probability"]
        return cls(
            ground_to_air_bandwidth_mhz=bandwidth["ground_to_air_mhz"],
            air_to_air_bandwidth_mhz=bandwidth["air_to_air_mhz"],
            air_to_ground_bandwidth_mhz=bandwidth["air_to_ground_mhz"],
            reference_channel_gain=propagation["reference_channel_gain"],
            nlos_attenuation_factor=propagation["nlos_attenuation_factor"],
            noise_power_w=propagation["noise_power_w"],
            los_alpha=los_probability["alpha"],
            los_beta=los_probability["beta"],
        )
