"""method 侧的在线策略调用循环；环境只负责真实事件推进。"""

from collections.abc import Callable, Mapping, Sequence

from env.contracts import PlacementDecision
from env.runtime.event_runtime import SchedulingRuntime
from env.types import EntityRef, TaskKey


def run_online_schedule(
    runtime: SchedulingRuntime,
    member_orders: Mapping[EntityRef, Sequence[TaskKey]],
    choose_placement: Callable[[TaskKey], PlacementDecision],
) -> None:
    """对已登记的 DAG 按 Member 序列逐次观测/决策，直到完成或截止。

    choose_placement 可在调用时读取当前观测；同批选择全部结束后才提交，
    确保多个同时可决策 Member 面对同一个环境时刻和资源状态。
    """
    orders = {member.index: iter(order) for member, order in member_orders.items()}
    ready = runtime.ready_member_ids
    while ready:
        current = [next(orders[member_id]) for member_id in ready]
        placements = {key: choose_placement(key) for key in current}
        runtime.submit_placements(placements)
        ready = runtime.advance_to_decision()
