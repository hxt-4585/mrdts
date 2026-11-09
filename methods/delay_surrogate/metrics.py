"""纯时延目标和执行诊断汇总；能耗/队列只记录，不作为目标。"""


class Summary:
    def __init__(self):
        self.slots = 0
        self.totals = dict.fromkeys(('dag_count', 'failed_dags', 'delay_sum_s', 'total_energy_j',
            'delay_cost', 'mean_energy_queue_j', 'planning_s', 'model_update_s', 'flight_violations'), 0.)
        self.max_queue = 0.

    def add(self, row):
        self.slots += 1
        for key in self.totals:
            self.totals[key] += row[key]
        self.max_queue = max(self.max_queue, row['max_energy_queue_j'])

    def result(self):
        t = self.totals
        return dict(slots_completed=self.slots, dag_count=int(t['dag_count']),
            failed_dags=int(t['failed_dags']), mean_delay_s=t['delay_sum_s'] / t['dag_count'],
            failure_rate=t['failed_dags'] / t['dag_count'],
            delay_cost_mean=t['delay_cost'] / self.slots,
            total_energy_j=t['total_energy_j'], mean_slot_energy_j=t['total_energy_j'] / self.slots,
            mean_energy_queue_j=t['mean_energy_queue_j'] / self.slots, max_energy_queue_j=self.max_queue,
            planning_mean_s=t['planning_s'] / self.slots,
            model_update_mean_s=t['model_update_s'] / self.slots,
            flight_violations=int(t['flight_violations']))
