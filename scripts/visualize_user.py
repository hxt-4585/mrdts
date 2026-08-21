"""可视化默认配置下的地面用户分布。

用法：
    python scripts/visualize_user.py

会在 scripts/output/ 目录下生成：
- user_distribution_default.png  默认区域划分上的用户二维散点图
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 将项目根目录加入 sys.path，便于导入 env 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.region import Region
from env.user import User


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def draw_user_distribution(region, users, ax):
    """在区域地图上绘制用户位置，并标注每个区域的用户数。"""
    config = region.config
    side_length = config.side_length
    region_map = region.region_map
    if region_map is None:
        raise RuntimeError("区域地图尚未生成，请先调用 Region.generate()")
    if users.positions is None:
        raise RuntimeError("用户尚未生成，请先调用 User.generate_from_region()")

    ax.imshow(
        region_map,
        cmap="Pastel1",
        vmin=0.5,
        vmax=config.region_count + 0.5,
        origin="lower",
        extent=(0, side_length, 0, side_length),
        interpolation="none",
        alpha=0.45,
        aspect="equal",
    )
    scatter = ax.scatter(
        users.positions[:, 0],
        users.positions[:, 1],
        c=users.region_ids,
        cmap="tab10",
        vmin=1,
        vmax=config.region_count,
        s=28,
        edgecolors="black",
        linewidths=0.35,
        label="ground user",
    )

    cell_size = config.cell_size
    for region_id, user_count in enumerate(users.region_user_counts, start=1):
        rows, columns = np.where(region_map == region_id)
        x = (columns.mean() + 0.5) * cell_size
        y = (rows.mean() + 0.5) * cell_size
        ax.text(
            x,
            y,
            f"R{region_id}\n{user_count} users",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.75},
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, side_length)
    ax.set_ylim(0, side_length)
    ax.set_title(f"Ground user distribution (total={users.num_users})")
    return scatter


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    region = Region()
    region.generate()
    users = User()
    users.generate_from_region(region)

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_user_distribution(region, users, ax)
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "user_distribution_default.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
