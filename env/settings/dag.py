"""DAG 生成器的 TOML 配置。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DAGConfig:
    """DAG 拓扑、特征与独立随机流的配置。"""

    n: int
    max_out: int
    rho: float
    delta: float
    seed: int | None
    input_data_range: tuple[int, int]
    cpu_range: tuple[float, float]
    intermediate_data_range: tuple[int, int]

    @classmethod
    def default(cls) -> "DAGConfig":
        """加载项目默认 DAG 配置。"""
        return cls.from_toml(Path(__file__).resolve().parents[2] / "config" / "dag.toml")

    @classmethod
    def from_toml(cls, path: str | Path) -> "DAGConfig":
        """从 DAG TOML 文件读取配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)
        topology = data["topology"]
        features = data["features"]
        return cls(
            n=topology["n"],
            max_out=topology["max_out"],
            rho=topology["rho"],
            delta=topology["delta"],
            seed=topology.get("seed"),
            input_data_range=tuple(features["input_data_range"]),
            cpu_range=tuple(features["cpu_range"]),
            intermediate_data_range=tuple(features["intermediate_data_range"]),
        )
