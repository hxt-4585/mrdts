"""方法无关的评价指标；失败 DAG 不从总数或截断时延代价中消失。"""


def slot_metrics(outcome, episode, slot):
    result = outcome.result
    count = len(result.dags)
    failures = sum(dag.failed for dag in result.dags)
    delays = [dag.completion_time - result.slot_start for dag in result.dags if not dag.failed]
    window = result.slot_end - result.slot_start
    return dict(episode=episode, slot=slot, dag_count=count, failed_dags=failures,
                failure_rate=failures / count if count else 0.0,
                successful_delay_sum_s=sum(delays),
                successful_mean_delay_s=sum(delays) / len(delays) if delays else None,
                truncated_delay_sum_s=sum(delays) + failures * window,
                compute_energy_j=result.compute_energy_j, tx_energy_j=result.tx_energy_j,
                flight_violations=outcome.flight_violations,
                flight_resamples=outcome.flight_resamples, flight_fallbacks=outcome.flight_fallbacks)


def summarize(rows):
    count = sum(row["dag_count"] for row in rows)
    failed = sum(row["failed_dags"] for row in rows)
    finished = count - failed
    return dict(dag_count=count, failed_dags=failed,
                failure_rate=failed / count if count else 0.0,
                successful_mean_delay_s=(sum(row["successful_delay_sum_s"] for row in rows) / finished
                                         if finished else None),
                truncated_mean_delay_s=(sum(row["truncated_delay_sum_s"] for row in rows) / count
                                       if count else None),
                compute_energy_j=sum(row["compute_energy_j"] for row in rows),
                tx_energy_j=sum(row["tx_energy_j"] for row in rows),
                flight_violations=sum(row["flight_violations"] for row in rows),
                flight_resamples=sum(row["flight_resamples"] for row in rows),
                flight_fallbacks=sum(row["flight_fallbacks"] for row in rows))
