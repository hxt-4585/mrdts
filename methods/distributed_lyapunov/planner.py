"""Queue-conditioned regional surrogates and synchronous one-hop projection.

Only a region's visible telemetry, users, proposals and partial derivatives
enter its model. No global simulator rollout is used in the optimizer.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Frame:
    xy: np.ndarray
    q: np.ndarray
    visible: np.ndarray
    owned: np.ndarray
    users: np.ndarray
    neighbors: tuple
    step: float


def frames(env):
    n = env.members.member_uav_count
    regions = env.members.region_ids[:n]
    mapping = {int(region): i for i, region in enumerate(env.masters.region_ids)}
    result = []
    for owner, region in enumerate(env.masters.region_ids):
        neighbors = env.masters.neighbor_region_ids[int(region)]
        telemetry = env.observe_master(owner).reshape(n, 3).astype(np.float32)
        result.append(Frame(telemetry[:, :2].copy(), telemetry[:, 2].copy(),
                            np.isin(regions, (region, *neighbors)), regions == region,
                            (env.users.positions[env.users.region_ids == region, :2] /
                             env.region.config.side_length).astype(np.float32),
                            tuple(mapping[int(r)] for r in neighbors),
                            env.members.config.max_horizontal_speed * env.env_config.flight_duration_s /
                            env.region.config.side_length))
    return result


class Proxy(nn.Module):
    def __init__(self, frame):
        super().__init__()
        self.register_buffer('users', torch.from_numpy(frame.users))
        width = 9 * len(frame.xy) + len(frame.xy) * len(frame.users)
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 64),
                                 nn.SiLU(), nn.Linear(64, 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        self.memory = []
        self.cursor = 0

    def features(self, frame, action):
        x = torch.as_tensor(frame.xy)
        visible = torch.as_tensor(frame.visible, dtype=torch.float32)
        owned = torch.as_tensor(frame.owned, dtype=torch.float32)
        a = action * visible[:, None]
        endpoint = (x + a * frame.step) * visible[:, None]
        q = torch.as_tensor(frame.q).log1p()
        distances = ((endpoint[:, None] - self.users[None])**2).sum(-1) * visible[:, None]
        return torch.cat((x.flatten(), endpoint.flatten(), a.flatten(), q, visible, owned, distances.flatten()))

    def cost(self, frame, action):
        return F.softplus(self.net(self.features(frame, action))).squeeze()

    def remember(self, frame, action, cost):
        item = self.features(frame, torch.as_tensor(action)).detach(), float(cost)
        if len(self.memory) < 4096:
            self.memory.append(item)
        else:
            self.memory[self.cursor] = item
        self.cursor = (self.cursor + 1) % 4096

    def fit(self):
        ids = torch.randint(len(self.memory), (min(64, len(self.memory)),)).tolist()
        x = torch.stack([self.memory[i][0] for i in ids])
        y = torch.tensor([self.memory[i][1] for i in ids])
        prediction = F.softplus(self.net(x).flatten())
        loss = F.mse_loss(prediction.log1p(), y.log1p())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())


class Controller:
    def __init__(self, initial_frames, seed):
        self.models = [Proxy(frame) for frame in initial_frames]
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def actions(self, current, exploration=0.):
        initial = np.zeros_like(current[0].xy)
        for frame in current:
            ids = np.flatnonzero(frame.owned)
            if len(ids):
                centers = frame.xy[ids].copy()
                for _ in range(4):
                    labels = ((frame.users[:, None] - centers[None])**2).sum(-1).argmin(1)
                    for j in range(len(ids)):
                        assigned = frame.users[labels == j]
                        if len(assigned):
                            centers[j] = assigned.mean(0)
                initial[ids] = np.clip((centers - frame.xy[ids])/frame.step, -1., 1.)
        initial *= .5
        if exploration:
            noise = self.rng.uniform(-1., 1., initial.shape).astype(np.float32)
            initial = (1-exploration) * initial + exploration * noise
        plans = []
        for frame in current:
            ids = np.flatnonzero(frame.owned)
            lower = np.maximum(-1., (1e-6 - frame.xy[ids])/frame.step)
            upper = np.minimum(1., (1.-1e-6-frame.xy[ids])/frame.step)
            plans.append([ids, np.clip(initial[ids], lower, upper), lower, upper])
        payload = 0
        if min(len(model.memory) for model in self.models) >= 64:
            for _ in range(3):
                gradients = [np.zeros_like(plan[1]) for plan in plans]
                for owner, (frame, model) in enumerate(zip(current, self.models)):
                    action = np.zeros_like(initial)
                    for sender in (owner, *frame.neighbors):
                        ids, proposal = plans[sender][:2]
                        action[ids] = proposal * frame.visible[ids, None]
                        if sender != owner:
                            payload += proposal.size
                    point = torch.tensor(action, requires_grad=True)
                    gradient = torch.autograd.grad(model.cost(frame, point) * len(frame.users), point)[0].numpy()
                    for target in (owner, *frame.neighbors):
                        ids = plans[target][0]
                        gradients[target] += gradient[ids] * frame.visible[ids, None]
                        if target != owner:
                            payload += 2 * len(ids)
                for plan, gradient in zip(plans, gradients):
                    plan[1] = np.clip(plan[1] - .2 * gradient / max(1., float(np.linalg.norm(gradient))),
                                      plan[2], plan[3])
        actions = np.zeros_like(initial)
        for ids, proposal, _, _ in plans:
            actions[ids] = proposal
        payload += sum(2 * int(frame.owned.sum()) * len(frame.neighbors) for frame in current)
        return actions.astype(np.float32), payload

    def feedback(self, current, actions, costs, v):
        self.steps += 1
        losses = []
        for frame, model, cost in zip(current, self.models, costs):
            model.remember(frame, actions, cost / (v * len(frame.users)))
            if self.steps % 4 == 0:
                losses.append(model.fit())
        return float(np.mean(losses)) if losses else 0.
