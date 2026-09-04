"""可视化默认场景中地面用户与 UAV 的初始分布。

用法：
    python scripts/visualize_uav.py

输出：
    scripts/output/uav_initial_distribution.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def draw_uav_distribution(region, users, masters, members, ax):
    """绘制区域、用户与两类 UAV 的初始二维位置。"""
    if region.region_map is None:
        raise RuntimeError("区域地图尚未生成，请先调用 Region.generate()")
    if any(entity.positions is None for entity in (users, masters, members)):
        raise RuntimeError("用户与 UAV 必须先完成生成")

    config = region.config
    side_length = config.side_length
    ax.imshow(
        region.region_map,
        cmap="Pastel1",
        vmin=0.5,
        vmax=config.region_count + 0.5,
        origin="lower",
        extent=(0, side_length, 0, side_length),
        interpolation="none",
        alpha=0.5,
        aspect="equal",
    )
    users_artist = ax.scatter(
        users.positions[:, 0],
        users.positions[:, 1],
        c=users.region_ids,
        cmap="tab10",
        vmin=1,
        vmax=config.region_count,
        s=20,
        alpha=0.7,
        linewidths=0,
        label="Ground users",
    )
    members_artist = ax.scatter(
        members.positions[: members.bs_index, 0],
        members.positions[: members.bs_index, 1],
        marker="^",
        s=95,
        c="#1565c0",
        edgecolors="white",
        linewidths=0.8,
        label="Member UAV",
        zorder=3,
    )
    bs_artist = ax.scatter(
        members.positions[members.bs_index : members.bs_index + 1, 0],
        members.positions[members.bs_index : members.bs_index + 1, 1],
        marker="s",
        s=125,
        c="#2e7d32",
        edgecolors="white",
        linewidths=0.9,
        label="BS",
        zorder=4,
    )
    masters_artist = ax.scatter(
        masters.positions[:, 0],
        masters.positions[:, 1],
        marker="X",
        s=150,
        c="#c62828",
        edgecolors="white",
        linewidths=0.9,
        label="Master UAV",
        zorder=5,
    )

    for region_id, member_count in enumerate(members.region_member_counts, start=1):
        rows, columns = (region.region_map == region_id).nonzero()
        x = (columns.mean() + 0.5) * config.cell_size
        y = (rows.mean() + 0.5) * config.cell_size
        ax.text(
            x,
            y,
            f"R{region_id}: {member_count} M",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.72},
            zorder=6,
        )

    ax.set_title(
        "Initial distribution: ground users, UAVs and BS "
        f"(Master/Member = {masters.config.master_altitude:g} m, "
        f"BS = {members.config.bs_altitude:g} m)"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, side_length)
    ax.set_ylim(0, side_length)
    ax.legend(loc="upper right")
    return {
        "users": users_artist,
        "masters": masters_artist,
        "members": members_artist,
        "bs": bs_artist,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    region = Region()
    region.generate()
    users = User()
    users.generate_from_region(region)
    masters = MasterUAV()
    masters.generate_from_region(region)
    members = MemberUAV()
    members.generate_from_region_and_user(region, users, masters)

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_uav_distribution(region, users, masters, members, ax)
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "uav_initial_distribution.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
