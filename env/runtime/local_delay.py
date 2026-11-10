"""Member 剩余任务全部本地执行时的轻量时延推演。"""

from collections import defaultdict
from typing import TYPE_CHECKING

from env.runtime.dag_runtime import topological_order
from env.runtime.server_queue import ServerState

if TYPE_CHECKING:
    from env.runtime.event_runtime import SchedulingRuntime
    from env.types import TaskKey


class LocalDelayEstimator:
    """按 DAG key、DAG 内拓扑序串行推演；不推进真实运行时。

    保留已完成任务的位置和完成时刻，复制地面服务器当前的核心及 FIFO
    负载。未来前驱回传取各路由独占资源参考时延的最大值，不预测未来
    其他 Member 的动作、共享带宽竞争或回传路径间的排队。
    """

    def __init__(self, runtime: "SchedulingRuntime"):
        self.runtime = runtime
        self.requests = defaultdict(list)
        self.orders = defaultdict(list)
        for request in sorted(runtime._requests.values(), key=lambda item: item.key):
            if request.owner_member is None:
                continue
            owner = request.owner_member.index
            self.requests[owner].append(request)
            keys = request.task_keys
            self.orders[owner].extend(keys[node] for node in topological_order(request.dag))

    @staticmethod
    def _compute_finish(available, frequencies, ready_at, cycles):
        core = min(range(len(available)), key=lambda i: (max(available[i], ready_at), i))
        finish_at = max(available[core], ready_at) + cycles / frequencies[core]
        available[core] = finish_at
        return finish_at

    def mean_delay(self, owner: int, first: "TaskKey | None" = None) -> float:
        """从当前时刻推演 owner 平均 DAG 完成时延（秒）。

        first 用于决策前的本地对照，强制先在本地执行本次选中的任务；
        完成后调用时所有已完成任务均保留真实结果。两边采用同一后续顺序。
        """
        runtime = self.runtime
        requests = self.requests[owner]
        available = {}
        for request in requests:
            ground = request.ground_device
            if ground in available:
                continue
            server = runtime.servers[ground]
            at = runtime.now
            if server.queue:
                # 仅复制服务器负载，复用真实 FIFO 派发与同批空闲核心规则。
                server = ServerState(server.node, server.core_frequencies, server.capacitance_factor,
                                     queue=server.queue.copy(), running=server.running.copy())
                while server.queue:
                    for core, record in tuple(server.running.items()):
                        if record.finish_at <= at:
                            server.finish(core, at)
                    server.dispatch(at)
                    if server.queue:
                        at = min(record.finish_at for record in server.running.values())
            cores = [server.running[i].finish_at if i in server.running else at
                     for i in range(len(server.core_frequencies))]
            available[ground] = cores

        locations = {}
        completion = dict.fromkeys((request.key for request in requests), runtime.slot_start)
        pending = []
        for key in self.orders[owner]:
            task = runtime.tasks[key]
            if task.compute_finish_at is not None:
                locations[key] = task.execution_node
                dag_key = (key.owner_member_id, key.user_id, key.dag_id)
                completion[dag_key] = max(completion[dag_key], task.compute_finish_at)
            elif key != first:
                pending.append(key)
        if first is not None:
            pending.insert(0, first)

        cursor = runtime.now
        for key in pending:
            task = runtime.tasks[key]
            dag_key = (key.owner_member_id, key.user_id, key.dag_id)
            request = runtime._requests[dag_key]
            ground = request.ground_device
            transfer_s = 0.0
            for parent in sorted(task.predecessors):
                route = runtime.route_planner.predecessor_route(
                    locations[parent], ground, ground, request.owner_member)
                bits = float(request.dag.edge_features[parent.node_id, key.node_id]) * 1000.
                transfer_s = max(transfer_s, sum(runtime.transfer_duration_s(hop, bits)
                                                for hop in route.hops))
            cursor = self._compute_finish(available[ground],
                                          runtime.servers[ground].core_frequencies,
                                          cursor + transfer_s, task.cpu_cycles)
            locations[key] = ground
            completion[dag_key] = max(completion[dag_key], cursor)
        return sum(at - runtime.slot_start for at in completion.values()) / len(requests)
