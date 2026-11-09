"""冻结 Member 的确定性 ERS 调度及执行后遥测；候选搜索不调用本模块。"""

from dataclasses import dataclass

import numpy as np
import torch

from env.contracts import PlacementDecision
from methods.hmappo.ers import ERS
from methods.hmappo.support import slot_metrics


@torch.inference_mode()
def execute_slot(adapter, members, actions):
    """与 HMAPPO 的 ERS + masked argmax 相同；不构建训练轨迹或调用 critic。"""
    env = adapter.env
    begin = env.begin_step(actions)
    plan = ERS(env._runtime).plan(begin.requests)
    orders = {node.index: iter(order) for node, order in plan.member_orders.items()}
    ready = begin.ready_member_ids
    decisions = 0
    device = next(members.parameters()).device
    while ready:
        keys = [next(orders[owner]) for owner in ready]
        observations, _, masks = adapter.member_inputs(keys)
        observations = torch.as_tensor(observations, device=device)
        masks = torch.as_tensor(masks, device=device)
        placements = {}
        for row, key in enumerate(keys):
            logits = members[key.owner_member_id].actor(observations[row])
            action = int(logits.masked_fill(~masks[row], -torch.inf).argmax())
            placements[key] = PlacementDecision(adapter.execution_node(action, key))
        decisions += len(keys)
        ready = env.submit_schedule(placements)
    result = env.end_step()
    row = slot_metrics(result.slot_result, env.epoch_index, env.step_index - 1,
                       int(begin.flight_violations.sum()))
    row.update(member_decisions=decisions, total_energy_j=float(result.total_energy_j.sum()),
               mean_energy_queue_j=float(result.energy_virtual_queue_j.mean()),
               max_energy_queue_j=float(result.energy_virtual_queue_j.max()))
    return result, row


def regional_delays(result, user_regions, regions, duration):
    """按 GD 固定区域分发执行后时延摘要，含无 Member 区域的本地执行 DAG。"""
    delays = np.zeros(regions)
    for dag in result.slot_result.dags:
        delays[user_regions[dag.key[1]] - 1] += dag.completion_time - result.slot_result.slot_start
    return delays / duration


def physical_target(view, result, delay, energy_budget):
    # 飞行已知、单独计算；其余能耗按 UAV ID 计账，跨区后仍回传给飞行前的负责人。
    residual = (result.total_energy_j - result.flight_energy_j) / energy_budget
    return np.concatenate(([delay], residual * view.owned))


def lyapunov_cost(views, result, delays, energy_budget, delay_weight):
    queues = sum((view.queues * view.owned for view in views))
    return float(queues @ (result.total_energy_j / energy_budget) + delay_weight * delays.sum())


@dataclass
class Summary:
    slots: int = 0
    dags: int = 0
    failed: int = 0
    delay: float = 0.
    energy: float = 0.
    cost: float = 0.
    queue: float = 0.
    max_queue: float = 0.
    planning: float = 0.
    updating: float = 0.
    violations: int = 0

    def add(self, row):
        self.slots += 1
        self.dags += row['dag_count']
        self.failed += row['failed_dags']
        self.delay += row['delay_sum_s']
        self.energy += row['total_energy_j']
        self.cost += row['lyapunov_cost']
        self.queue += row['mean_energy_queue_j']
        self.max_queue = max(self.max_queue, row['max_energy_queue_j'])
        self.planning += row['planning_s']
        self.updating += row['model_update_s']
        self.violations += row['flight_violations']

    def result(self):
        return dict(slots_completed=self.slots, dag_count=self.dags, failed_dags=self.failed,
                    mean_delay_s=self.delay / self.dags, failure_rate=self.failed / self.dags,
                    total_energy_j=self.energy, mean_slot_energy_j=self.energy / self.slots,
                    lyapunov_cost_mean=self.cost / self.slots,
                    mean_energy_queue_j=self.queue / self.slots, max_energy_queue_j=self.max_queue,
                    planning_mean_s=self.planning / self.slots,
                    model_update_mean_s=self.updating / self.slots, flight_violations=self.violations)
