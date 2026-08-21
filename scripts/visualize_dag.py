"""可视化 DAG 示例。

用法：
    python scripts/visualize_dag.py

会在 scripts/output/ 目录下生成若干张 DAG 拓扑图：
- dag_default.png           默认参数的单个 DAG（含节点/边特征标注）
- dag_rho_comparison.png    不同 rho 取值下的 DAG 拓扑对比
"""

import os
import sys
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # 无界面后端，便于脚本化出图
import matplotlib.pyplot as plt
import networkx as nx

# 将项目根目录加入 sys.path，便于导入 env 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.settings import DAGConfig
from env.graph_utils import DAGGenerator

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _build_graph(dag):
    """根据 DAG 构建 networkx 有向图。"""
    g = nx.DiGraph()
    g.add_nodes_from(range(dag.node_num))
    g.add_edges_from(dag.edges)
    return g


def _layered_pos(dag):
    """按拓扑分层生成布局坐标：层号作 x 轴，层内节点序号作 y 轴。"""
    g = _build_graph(dag)
    pos = {}
    for layer_idx, layer in enumerate(nx.topological_generations(g)):
        for j, node in enumerate(sorted(layer)):
            pos[node] = (layer_idx, -j)
    return g, pos


def draw_single(dag, title, path):
    """绘制带节点/边特征标注的单个 DAG。"""
    g, pos = _layered_pos(dag)
    fig, ax = plt.subplots(figsize=(9, 6.5))

    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color="#7fb3d5", node_size=1500, edgecolors="black", linewidths=1
    )
    nx.draw_networkx_edges(
        g, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=16, node_size=1500
    )

    # 节点标签：编号 + 输入数据量 + 计算量
    labels = {}
    for i in range(dag.node_num):
        data = int(round(dag.node_features[i, 0]))
        cpu = dag.node_features[i, 1]
        labels[i] = f"{i}\nd={data}KB\nc={cpu:.1e}"
    nx.draw_networkx_labels(g, pos, ax=ax, labels=labels, font_size=7)

    # 边标签：中间结果数据量 (KB)
    edge_labels = {e: str(dag.edge_features[e]) for e in dag.edges}
    nx.draw_networkx_edge_labels(
        g, pos, ax=ax, edge_labels=edge_labels, font_size=6, font_color="crimson"
    )

    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_rho_comparison(rho_list, path):
    """对比不同 rho 取值下的 DAG 拓扑（仅展示结构）。"""
    cols = 2
    rows = (len(rho_list) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, rho in zip(axes, rho_list):
        cfg = replace(DAGConfig.default(), rho=rho)
        dag = DAGGenerator(cfg).generate_single_dag()
        g, pos = _layered_pos(dag)

        nx.draw_networkx_nodes(
            g, pos, ax=ax, node_color="#7fb3d5", node_size=600, edgecolors="black"
        )
        nx.draw_networkx_edges(
            g, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12, node_size=600
        )
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
        ax.set_title(f"rho={rho}, edges={len(dag.edges)}", fontsize=11)
        ax.axis("off")

    # 隐藏多余的空子图
    for ax in axes[len(rho_list):]:
        ax.axis("off")

    fig.suptitle("DAG topology under different rho (n=10, delta=0.5)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 默认参数的单个 DAG（含特征）
    default_cfg = DAGConfig.default()
    dag = DAGGenerator(default_cfg).generate_single_dag()
    draw_single(
        dag,
        f"DAG example (n={default_cfg.n}, max_out={default_cfg.max_out}, "
        f"rho={default_cfg.rho}, delta={default_cfg.delta})",
        os.path.join(OUTPUT_DIR, "dag_default.png"),
    )
    print("已生成: dag_default.png")

    # 2. 不同 rho 的拓扑对比
    draw_rho_comparison([0.5, 1.0, 1.5, 2.0], os.path.join(OUTPUT_DIR, "dag_rho_comparison.png"))
    print("已生成: dag_rho_comparison.png")


if __name__ == "__main__":
    main()
