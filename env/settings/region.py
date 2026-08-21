"""区域划分的 TOML 配置。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class RegionConfig:
    """正方形地图与区域划分参数。"""

    side_length: float
    grid_size: int
    region_count: int
    min_area_ratio: float
    area_imbalance: float
    lloyd_iterations: int
    capacity_iterations: int
    seed: int | None

    @classmethod
    def default(cls) -> "RegionConfig":
        """加载项目默认区域配置。"""
        return cls.from_toml(Path(__file__).resolve().parents[2] / "config" / "region.toml")

    @property
    def cell_size(self) -> float:
        """基础网格的边长（m）。"""
        return self.side_length / self.grid_size

    @property
    def total_cells(self) -> int:
        """基础网格总数。"""
        return self.grid_size ** 2

    @property
    def min_cells_per_region(self) -> int:
        """每个区域的最小网格数。"""
        return int(self.total_cells * self.min_area_ratio)

    @classmethod
    def from_toml(cls, path: str | Path) -> "RegionConfig":
        """从区域 TOML 文件读取配置。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)
        geometry = data["geometry"]
        partition = data["partition"]
        return cls(
            side_length=geometry["side_length"],
            grid_size=geometry["grid_size"],
            region_count=partition["region_count"],
            min_area_ratio=partition["min_area_ratio"],
            area_imbalance=partition["area_imbalance"],
            lloyd_iterations=partition["lloyd_iterations"],
            capacity_iterations=partition["capacity_iterations"],
            seed=partition.get("seed"),
        )
