"""Block proximal SCA of a frozen-traffic analytic communication-delay proxy.

Each factor holds only its region's users/traffic and one-hop UAV positions.
Neighbor oracles return a scalar cost and derivatives for the requesting block.
The quadratic subproblem has a box-projection solution. Bounded backtracking
accepts only a valid quadratic upper model at the proposed point; otherwise the
block stays put. No neural network, fitting, simulator, or future workload is used.
"""

from dataclasses import dataclass, field

import numpy as np

from env.settings import ChannelConfig


@dataclass
class LocalProblem:
    member_ids: np.ndarray
    owned_ids: np.ndarray
    positions: np.ndarray  # visible moving UAVs, then local users and fixed BS
    sources: np.ndarray
    targets: np.ndarray
    weighted_bits: np.ndarray
    powers_w: np.ndarray
    air_to_air: np.ndarray
    channel: ChannelConfig
    smoothing_m: float
    rows: dict[int, int] = field(init=False)

    def __post_init__(self):
        self.rows = {int(member): row for row, member in enumerate(self.member_ids)}
        self.sources = np.asarray(self.sources, dtype=np.intp)
        self.targets = np.asarray(self.targets, dtype=np.intp)
        self.air_to_air = np.asarray(self.air_to_air, dtype=bool)

    def evaluate(self):
        """Return weighted link delay (s) and exact analytic position gradient.

        A 0.1 m smoothed horizontal norm removes the vertical-link cusp of the
        elevation-dependent LoS model. This is a declared model approximation.
        Effective bit weights incorporate previously observed resource sharing.
        """
        delta = self.positions[self.sources] - self.positions[self.targets]
        horizontal = np.sqrt(np.sum(delta[:, :2] ** 2, axis=1) + self.smoothing_m**2)
        height = np.abs(delta[:, 2])
        distance2 = horizontal**2 + height**2
        angle = np.arctan2(height, horizontal) * (180. / np.pi)
        c = self.channel
        los = 1. / (1. + c.los_alpha * np.exp(-c.los_beta * (angle - c.los_alpha)))
        dlos = -los * (1. - los) * c.los_beta * (180. / np.pi) * height / distance2
        los[self.air_to_air] = 1.
        dlos[self.air_to_air] = 0.
        mixing = c.nlos_attenuation_factor + (1. - c.nlos_attenuation_factor) * los
        gain = c.reference_channel_gain * mixing / distance2
        dgain = c.reference_channel_gain * (
            (1. - c.nlos_attenuation_factor) * dlos / distance2
            - 2. * horizontal * mixing / distance2**2)
        bandwidth = c.total_bandwidth_mhz * 1e6
        signal_scale = self.powers_w / (c.noise_density_w_per_hz * bandwidth)
        snr = signal_scale * gain
        log_rate = np.log1p(snr)
        coefficient = self.weighted_bits * np.log(2.) / bandwidth
        cost = float(np.sum(coefficient / log_rate))
        radial_gradient = -coefficient * signal_scale * dgain / ((1. + snr) * log_rate**2)
        edge_gradient = (radial_gradient / horizontal)[:, None] * delta[:, :2]
        gradient = np.zeros((len(self.positions), 2))
        np.add.at(gradient, self.sources, edge_gradient)
        np.add.at(gradient, self.targets, -edge_gradient)
        return cost, gradient[:len(self.member_ids)]

    def block_oracle(self, ids, xy):
        """Evaluate a neighbor-owned proposal without exposing this factor's data."""
        rows = np.array([self.rows[int(member)] for member in ids], dtype=np.intp)
        previous = self.positions[rows, :2].copy()
        self.positions[rows, :2] = xy
        value, gradient = self.evaluate()
        self.positions[rows, :2] = previous
        return value, gradient[rows]

    def receive(self, ids, xy):
        rows = [self.rows[int(member)] for member in ids]
        self.positions[rows, :2] = xy


def coordinate(problems, neighbors, step_m, side_m, rounds=3, backtracks=6):
    """Fixed-budget Gauss-Seidel block SCA, using only one-hop factor oracles.

    Blocks are serviced in public Master-ID order. This is distributed in data
    and computation, with sequential coordination rather than a claimed parallel
    speedup. All inner iterates obey the SAME current-slot flight box.
    """
    origins, lower, upper = [], [], []
    # Keep float32 movement arithmetic inside the physical closed scene box.
    margin = 4. * np.finfo(np.float32).eps * side_m
    for problem in problems:
        rows = [problem.rows[int(member)] for member in problem.owned_ids]
        origin = problem.positions[rows, :2].copy()
        origins.append(origin)
        lower.append(np.maximum(origin - step_m, margin))
        upper.append(np.minimum(origin + step_m, side_m - margin))
    metrics = dict(block_attempts=0, accepted_steps=0, oracle_evaluations=0,
                   neighbor_oracle_calls=0, backtracking_rejections=0,
                   proxy_before_s=sum(problem.evaluate()[0] for problem in problems))
    for _ in range(rounds):
        for master, problem in enumerate(problems):
            ids = problem.owned_ids
            if not len(ids):
                continue
            affected = (master, *neighbors[master])
            rows = [problem.rows[int(member)] for member in ids]
            current = problem.positions[rows, :2].copy()
            cost, gradient = 0., np.zeros_like(current)
            for other in affected:
                value, partial = problems[other].block_oracle(ids, current)
                cost += value
                gradient += partial
            metrics["block_attempts"] += 1
            metrics["oracle_evaluations"] += len(affected)
            metrics["neighbor_oracle_calls"] += len(affected) - 1
            curvature = max(float(np.linalg.norm(gradient, axis=1).max()) / step_m, 1e-12)
            for _ in range(backtracks):
                proposal = np.clip(current - gradient / curvature, lower[master], upper[master])
                trial = sum(problems[other].block_oracle(ids, proposal)[0] for other in affected)
                metrics["oracle_evaluations"] += len(affected)
                metrics["neighbor_oracle_calls"] += len(affected) - 1
                difference = proposal - current
                majorant = cost + float(np.sum(gradient * difference)) + (
                    .5 * curvature * float(np.sum(difference**2)))
                if trial <= majorant + 1e-14 and trial <= cost + 1e-14:
                    for other in affected:
                        problems[other].receive(ids, proposal)
                    metrics["accepted_steps"] += 1
                    break
                curvature *= 2.
                metrics["backtracking_rejections"] += 1
    actions = np.empty((sum(len(problem.owned_ids) for problem in problems), 2))
    for master, problem in enumerate(problems):
        rows = [problem.rows[int(member)] for member in problem.owned_ids]
        actions[problem.owned_ids] = np.clip(
            (problem.positions[rows, :2] - origins[master]) / step_m, -1., 1.)
    metrics["proxy_after_s"] = sum(problem.evaluate()[0] for problem in problems)
    return actions, metrics
