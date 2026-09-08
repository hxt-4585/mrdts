"""On-policy samples, with masks and observations frozen at sampling time."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Rollout:
    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    critic_observations: np.ndarray
    values: np.ndarray
    returns: np.ndarray
