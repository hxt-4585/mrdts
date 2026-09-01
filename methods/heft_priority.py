"""HEFT upward-rank 子任务优先级。"""

from env.dag_generator import DAG


class HEFTPriority:
    """只计算确定性优先级，不参与执行节点选择。"""

    @staticmethod
    def ranks(
        dag: DAG, average_compute_s: dict[int, float], average_edge_comm_s: dict[tuple[int, int], float]
    ) -> dict[int, float]:
        successors = {node_id: [] for node_id in range(dag.node_num)}
        for predecessor, successor in dag.edges:
            successors[predecessor].append(successor)
        values: dict[int, float] = {}

        def rank(node_id: int) -> float:
            if node_id in values:
                return values[node_id]
            tail = max(
                (
                    average_edge_comm_s[(node_id, successor)] + rank(successor)
                    for successor in successors[node_id]
                ),
                default=0.0,
            )
            values[node_id] = average_compute_s[node_id] + tail
            return values[node_id]

        return {node_id: rank(node_id) for node_id in range(dag.node_num)}

    @classmethod
    def order(
        cls, dag: DAG, average_compute_s: dict[int, float], average_edge_comm_s: dict[tuple[int, int], float]
    ) -> tuple[int, ...]:
        ranks = cls.ranks(dag, average_compute_s, average_edge_comm_s)
        return tuple(sorted(ranks, key=lambda node_id: (-ranks[node_id], node_id)))
