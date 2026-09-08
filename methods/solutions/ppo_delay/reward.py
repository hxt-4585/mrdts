"""Only the normalized global censored DAG delay enters reward."""
import numpy as np
from env.runtime.slot_result import SlotResult

def delay_metrics(result: SlotResult) -> dict[str, float]:
    """Censored mean DAG delay, counting every DAG equally (including failures)."""
    horizon = result.slot_end - result.slot_start
    if horizon <= 0 or not result.dags:
        raise ValueError('A positive slot duration and at least one DAG are required')
    delays = [horizon if dag.failed else dag.completion_time - result.slot_start
              for dag in result.dags]
    mean_delay = float(np.mean(np.clip(delays, 0., horizon)))
    return dict(mean_delay_s=mean_delay, reward=-mean_delay / horizon,
                failure_rate=sum(d.failed for d in result.dags) / len(result.dags),
                max_delay_s=float(max(delays)))


