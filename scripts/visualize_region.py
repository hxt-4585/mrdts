"""可视化区域地图。

用法：
    python scripts/visualize_region.py

会在 scripts/output/ 目录下生成：
- region_default.png      默认 R=4 的区域地图（标注各区域面积）
- region_comparison.png   不同 region_count 下的区域划分对比
- region_imbalance.png    不同 area_imbalance 下的面积偏差对比
"""

import os
import sys
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # 无界面后端，便于脚本化出图
import matplotlib.pyplot as plt
import numpy as np

# 将项目根目录加入 sys.path，便于导入 env 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.randomness import RandomStreams
from env.entities.region import Region
from env.settings import RegionConfig

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def draw_region_map(region_map, config, ax, title=None, show_grid=True):
    """绘制区域地图、区域编号，并可选地显示每个基础网格的边界。"""
    R = config.region_count
    ny, nx = region_map.shape

    cmap = matplotlib.colormaps.get_cmap("tab20")
    ax.imshow(region_map, cmap=cmap, vmin=0.5, vmax=R + 0.5, origin="lower", aspect="equal")

    # 各区域面积（网格数）
    sizes = np.bincount(region_map.ravel(), minlength=R + 1)[1:]

    # 在区域质心标注区域编号
    for r in range(1, R + 1):
        ys, xs = np.where(region_map == r)
        if len(xs) == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        ax.text(
            cx, cy, str(r), ha="center", va="center", fontsize=11,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55),
        )

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel("x grid index")
    ax.set_ylabel("y grid index")
    ax.set_xticks(np.arange(0, nx + 1, 10))
    ax.set_yticks(np.arange(0, ny + 1, 10))
    if show_grid:
        # imshow 的单元格中心在整数坐标，边界因而位于 n - 0.5。
        # 采用次刻度避免在坐标轴上显示 101 个刻度标签。
        ax.set_xticks(np.arange(-0.5, nx, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, ny, 1.0), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.25, alpha=0.22)
        ax.tick_params(which="minor", bottom=False, left=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 默认 R=4，固定种子复现
    cfg = RegionConfig.default()
    mgr = Region(cfg, rng=RandomStreams.from_config().region)
    region_map = mgr.generate()
    ok, sizes = mgr.validate()
    print(f"[默认] validate={ok}, 各区域面积={sizes.tolist()}, 下限={cfg.min_cells_per_region}")

    fig, ax = plt.subplots(figsize=(7, 7))
    title = (
        f"Default: R={cfg.region_count}\n"
        f"sizes={sizes.tolist()}, min={cfg.min_cells_per_region}"
    )
    draw_region_map(region_map, cfg, ax, title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "region_default.png"), dpi=150)
    plt.close(fig)
    print("已生成: region_default.png")

    # 2. 不同 region_count 的对比（统一用较小的面积下限保证参数可行）
    r_list = [2, 3, 4, 5, 6, 8]
    cols, rows = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 4.6))
    for ax, r in zip(axes.flatten(), r_list):
        c = replace(RegionConfig.default(), region_count=r, min_area_ratio=0.05)
        m = Region(c, rng=RandomStreams.from_config().region)
        rm = m.generate()
        ok, sz = m.validate()
        draw_region_map(
            rm, c, ax, f"R={r}, sizes={sz.tolist()}, min={c.min_cells_per_region}"
        )

    # 隐藏多余的空子图
    for ax in axes.flatten()[len(r_list):]:
        ax.axis("off")

    fig.suptitle("Region partition under different region_count", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "region_comparison.png"), dpi=150)
    plt.close(fig)
    print("已生成: region_comparison.png")

    # 3. 不同 area_imbalance 的面积偏差对比
    imb_list = [0.0, 0.3, 0.6, 0.8]
    cols, rows = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 4.6))
    for ax, imb in zip(axes.flatten(), imb_list):
        c = replace(RegionConfig.default(), area_imbalance=imb)
        m = Region(c, rng=RandomStreams.from_config().region)
        rm = m.generate()
        ok, sz = m.validate()
        draw_region_map(rm, c, ax, f"imbalance={imb}, sizes={sz.tolist()}")

    fig.suptitle("Region partition under different area_imbalance (R=4)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "region_imbalance.png"), dpi=150)
    plt.close(fig)
    print("已生成: region_imbalance.png")


if __name__ == "__main__":
    main()
