"""直接运行以创建 MRDTSEnv 场景并显示 UAV 分布。"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.env import MRDTSEnv
from env.settings import DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig


def create_environment() -> MRDTSEnv:
    """创建并生成一个完整的固定场景。"""
    settings = {
        "env": EnvConfig.default(),
        "dag": DAGConfig.default(),
        "region": RegionConfig.default(),
        "user": UserConfig.default(),
        "uav": UAVConfig.default(),
    }
    environment = MRDTSEnv(settings)
    environment.generate_scene()
    return environment


def show_distribution(environment: MRDTSEnv, *, show=True):
    """显示区域、Member UAV、Master UAV 与 BS 的二维分布。"""
    import matplotlib.pyplot as plt

    region = environment.region
    users = environment.users
    members = environment.members
    masters = environment.masters
    side_length = region.config.side_length

    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(
        region.region_map,
        cmap="Pastel1",
        vmin=0.5,
        vmax=region.config.region_count + 0.5,
        origin="lower",
        extent=(0, side_length, 0, side_length),
        interpolation="none",
        alpha=0.55,
        aspect="equal",
    )
    axis.scatter(
        users.positions[:, 0],
        users.positions[:, 1],
        c=users.region_ids,
        cmap="tab10",
        vmin=1,
        vmax=region.config.region_count,
        s=22,
        alpha=0.7,
        linewidths=0,
        label="Ground users",
        zorder=2,
    )
    axis.scatter(
        members.positions[:members.bs_index, 0],
        members.positions[:members.bs_index, 1],
        marker="^",
        s=100,
        c="#1565c0",
        edgecolors="white",
        linewidths=0.8,
        label="Member UAV",
        zorder=3,
    )
    axis.scatter(
        masters.positions[:, 0],
        masters.positions[:, 1],
        marker="X",
        s=160,
        c="#c62828",
        edgecolors="white",
        linewidths=0.9,
        label="Master UAV",
        zorder=4,
    )
    axis.scatter(
        members.positions[members.bs_index, 0],
        members.positions[members.bs_index, 1],
        marker="s",
        s=140,
        c="#2e7d32",
        edgecolors="white",
        linewidths=0.9,
        label="BS",
        zorder=5,
    )

    axis.set_title("Region, ground users, Master UAV, Member UAV and BS")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_xlim(0, side_length)
    axis.set_ylim(0, side_length)
    axis.legend()
    figure.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    environment = create_environment()
    show_distribution(environment)
