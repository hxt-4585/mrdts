"""DAG 生成可视化示例脚本。

生成若干组不同参数配置的 DAG 示例并保存为 PNG 图片，便于直观检查
DAG 生成逻辑（分层结构、依赖边、节点/边特征）。

用法：
    python scripts/visualize_dag.py            # 图片保存到 ./figs
    python scripts/visualize_dag.py 输出目录

依赖：matplotlib、networkx
"""

import os
import sys

# 保证从任意目录运行都能导入 env 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # 后台模式：只保存图片，不弹出窗口
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from env.config import DAGConfig
from env.graph_utils import DAGGenerator, TaskDAG

# 中文字体（Windows 自带微软雅黑/黑体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def layered_pos(layer_sizes, x_spacing=2.0, y_spacing=1.0):
    """按层计算节点坐标：层号映射到 x 轴，层内节点沿 y 轴均匀居中分布。

    与生成器内部"按层连续编号"的约定一致，保证绘图与拓扑一一对应。
    """
    pos = {}
    node = 0
    for layer_idx, size in enumerate(layer_sizes):
        for k in range(size):
            y = (k - (size - 1) / 2) * y_spacing
            pos[node] = (layer_idx * x_spacing, y)
            node += 1
    return pos


def draw_dag(dag: TaskDAG, title: str, filename: str) -> None:
    """绘制一个 DAG 并保存为图片。

    绘图约定：
    - 节点按层从左到右排列，颜色表示层号，大小正比于计算量；
    - 节点标签为"编号/输入数据量(KB)"；
    - 边标签为"中间结果数据量(KB)"。
    """
    topo = dag.topology
    G = nx.DiGraph()
    G.add_nodes_from(range(topo.node_num))
    G.add_edges_from(topo.edges)

    # 计算每个节点所属的层号
    layer_of = {}
    start = 0
    for layer_idx, size in enumerate(topo.layer_sizes):
        for node in range(start, start + size):
            layer_of[node] = layer_idx
        start += size

    num_layers = len(topo.layer_sizes)
    pos = layered_pos(topo.layer_sizes)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 节点大小正比于计算量（面积缩放）
    cpus = dag.node_cpu_cycles.flatten()
    cpu_norm = (cpus - cpus.min()) / (cpus.max() - cpus.min() + 1e-12)
    node_sizes = 250 + 2000 * cpu_norm

    cmap = plt.get_cmap("viridis")
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(range(topo.node_num)),
        node_color=[layer_of[n] for n in range(topo.node_num)],
        cmap=cmap, vmin=-0.5, vmax=num_layers - 0.5,
        node_size=node_sizes, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=True,
        arrowstyle="-|>", arrowsize=14,
        edge_color="gray", width=1.2,
    )

    # 节点标签：编号 + 输入数据量
    node_labels = {
        n: f"{n}\n{int(dag.node_data_sizes[n, 0])}KB"
        for n in range(topo.node_num)
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)

    # 边标签：中间结果数据量
    edge_labels = {e: str(dag.edge_data_sizes[e]) for e in topo.edges}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=7, ax=ax, rotate=False,
    )

    # 层号颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, num_layers - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=range(num_layers))
    cbar.set_label("层号")

    ax.set_title(title, fontsize=12)
    ax.text(
        0.5, -0.12,
        "节点标签: 编号/输入数据量(KB)；边标签: 中间结果数据量(KB)；"
        f"节点大小: 计算量；总输入 {dag.total_data_size:.0f} KB，"
        f"总计算量 {dag.total_cpu_cycles:.2e} cycles",
        transform=ax.transAxes, ha="center", fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: str = "figs") -> None:
    """生成多组参数配置下的 DAG 示例图片。"""
    os.makedirs(out_dir, exist_ok=True)
    examples = [
        ("默认参数：10 节点，最大出度 2，rho=1.0，delta=0.5",
         DAGConfig(),
         "dag_default.png"),
        ("深层 DAG：rho=0.5，层数更多（12 节点）",
         DAGConfig(num_subtasks=12, shape_parameter=0.5),
         "dag_deep.png"),
        ("扁平 DAG：rho=1.5，层数少（10 节点）",
         DAGConfig(num_subtasks=10, shape_parameter=1.5),
         "dag_flat.png"),
        ("链式 DAG：最大出度 1（10 节点）",
         DAGConfig(num_subtasks=10, max_out_degree=1),
         "dag_chain.png"),
        ("大 DAG：20 节点，最大出度 3，delta=1.0",
         DAGConfig(num_subtasks=20, max_out_degree=3, regularity_parameter=1.0),
         "dag_large.png"),
    ]
    for title, cfg, fname in examples:
        dag = DAGGenerator(cfg).generate_dag()
        path = os.path.join(out_dir, fname)
        draw_dag(dag, title, path)
        print(f"[已生成] {path}")
        print(f"   层数={len(dag.topology.layer_sizes)}，"
              f"各层节点数={dag.topology.layer_sizes}，"
              f"边数={len(dag.topology.edges)}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "figs"
    main(out_dir)
