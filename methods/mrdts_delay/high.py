"""Regional contributions to global mean delay and one-hop flight optimization."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class View:
    xy: np.ndarray
    visible: np.ndarray
    owned: np.ndarray
    users: np.ndarray
    adjacent: tuple[int, ...]
    stride: float


def regional_targets(slot, user_regions, region_count):
    """Include every DAG, including automatically locally executed DAGs."""
    labels = np.zeros(region_count, dtype=np.float64)
    denominator = len(slot.dags)
    for dag in slot.dags:
        labels[user_regions[dag.key[1]] - 1] += (dag.completion_time - slot.slot_start) / denominator
    return labels


class DelayModel(nn.Module):
    def __init__(self, count, config):
        super().__init__()
        width, hidden = count * 4, config['hidden']
        self.network = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['learning_rate'])
        self.replay_x = torch.empty(config['replay_size'], width)
        self.replay_y = torch.empty(config['replay_size'])
        self.size = self.next_index = 0

    def features(self, view, destination):
        points = destination.reshape(-1, len(view.xy), 2)
        visibility = torch.as_tensor(view.visible, dtype=points.dtype, device=points.device)
        ownership = torch.as_tensor(view.owned, dtype=points.dtype, device=points.device)
        return torch.cat((points.mul(visibility[None, :, None]).flatten(1),
                          visibility.expand(points.shape[0], -1),
                          ownership.expand(points.shape[0], -1)), dim=1)

    def forward(self, view, destination):
        return F.softplus(self.network(self.features(view, destination))).flatten()

    def remember(self, x, y):
        self.replay_x[self.next_index].copy_(x)
        self.replay_y[self.next_index] = float(y)
        self.next_index = (self.next_index + 1) % len(self.replay_y)
        self.size = min(self.size + 1, len(self.replay_y))

    def fit(self, config, generator):
        losses = []
        for _ in range(config['fit_steps']):
            indices = torch.randint(self.size, (min(self.size, config['batch_size']),), generator=generator)
            predictions = F.softplus(self.network(self.replay_x[indices])).flatten()
            loss = F.mse_loss(torch.log1p(predictions), torch.log1p(self.replay_y[indices]))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().item())
        return sum(losses) / len(losses)


class DelayPlanner:
    def __init__(self, env, config):
        self.config = config
        self.count = env.members.member_uav_count
        self.side = env.region.config.side_length
        self.stride = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s / self.side
        self.regions = tuple(int(r) for r in env.masters.region_ids)
        indices = {region: i for i, region in enumerate(self.regions)}
        self.adjacent = tuple(tuple(indices[r] for r in env.masters.neighbor_region_ids[region])
                              for region in self.regions)
        self.user_xy = tuple((env.users.positions[env.users.region_ids == region, :2] / self.side)
                             .astype(np.float32) for region in self.regions)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'] + 100)
            self.models = [DelayModel(self.count, config) for _ in self.regions]
        self.fit_rng = torch.Generator().manual_seed(config['seed'] + 101)
        self.observations = 0
        self.reset(0)

    def reset(self, epoch):
        self.random = [np.random.default_rng(np.random.SeedSequence([self.config['seed'], epoch, i]))
                       for i in range(len(self.regions))]
        self.previous = None
        self.goal = np.zeros((self.count, 2), dtype=np.float32)
        self.prior_owners = np.full(self.count, -1)
        self.exploring = np.zeros(len(self.regions), bool)
        self.tick = 0

    def views(self, env):
        member_regions = env.members.region_ids[:self.count]
        xy = env.members.positions[:self.count, :2] / self.side
        result = []
        for i, region in enumerate(self.regions):
            visible_regions = (region, *(self.regions[j] for j in self.adjacent[i]))
            visible = np.isin(member_regions, visible_regions)
            result.append(View(np.where(visible[:, None], xy, 0.).astype(np.float32), visible,
                               member_regions == region, self.user_xy[i], self.adjacent[i], self.stride))
        return result

    def coverage(self, views):
        action = np.zeros((self.count, 2), dtype=np.float32)
        for view in views:
            ids = np.flatnonzero(view.owned)
            if ids.size == 0:
                continue
            centers = view.xy[ids].copy()
            for _ in range(6):
                assignment = np.square(view.users[:, None] - centers).sum(2).argmin(1)
                for j in range(len(ids)):
                    group = view.users[assignment == j]
                    if group.size:
                        centers[j] = group.mean(0)
            action[ids] = np.clip((centers - view.xy[ids]) / view.stride, -1., 1.)
        return action

    def optimize(self, views, initial):
        groups = [np.flatnonzero(view.owned) for view in views]
        epsilon = 4 * np.finfo(np.float32).eps
        lower = [np.maximum(-v.stride, epsilon - v.xy[g]) for v, g in zip(views, groups)]
        upper = [np.minimum(v.stride, 1 - epsilon - v.xy[g]) for v, g in zip(views, groups)]
        shifts = [np.clip(initial[g] * v.stride, lo, hi)
                  for v, g, lo, hi in zip(views, groups, lower, upper)]
        for iteration in range(self.config['rounds']):
            gradients = [np.zeros_like(shift) for shift in shifts]
            for owner, (view, model) in enumerate(zip(views, self.models)):
                partners = (owner, *view.adjacent)
                proposed = view.xy.copy()
                for other in partners:
                    ids = groups[other]
                    proposed[ids] += shifts[other] * view.visible[ids, None]
                variable = torch.tensor(proposed, requires_grad=True)
                # Each output already uses the global denominator. No regional weighting.
                derivative = torch.autograd.grad(model(view, variable).sum(), variable)[0].numpy()
                for other in partners:
                    ids = groups[other]
                    gradients[other] += derivative[ids] * view.visible[ids, None]
            rate = self.config['gradient_rate'] / math.sqrt(iteration + 1)
            for j, derivative in enumerate(gradients):
                shifts[j] = np.clip(shifts[j] - rate * derivative / max(1., float(np.linalg.norm(derivative))),
                                    lower[j], upper[j])
        action = np.zeros_like(initial)
        for view, ids, shift in zip(views, groups, shifts):
            action[ids] = shift / view.stride
        return action

    def act(self, views, exploration):
        initial = self.coverage(views)
        if self.previous is not None:
            initial = (initial + self.previous) / 2
        if min(model.size for model in self.models) >= self.config['warmup_samples']:
            action = self.optimize(views, initial)
        else:
            action = initial
        if exploration:
            refresh = self.tick % self.config['exploration_hold'] == 0
            if refresh:
                self.prior_owners.fill(-1)
                self.exploring = np.array([rng.random() < exploration for rng in self.random])
            for owner, view in enumerate(views):
                ids = np.flatnonzero(view.owned)
                arrivals = ids[self.prior_owners[ids] != owner]
                rng = self.random[owner]
                chosen = view.users[rng.integers(len(view.users), size=len(arrivals))]
                self.goal[arrivals] = np.clip(chosen + rng.normal(0., .12, chosen.shape), .001, .999)
                self.prior_owners[ids] = owner
                if self.exploring[owner]:
                    action[ids] = np.clip((self.goal[ids] - view.xy[ids]) / view.stride, -1., 1.)
        self.tick += 1
        self.previous = action.copy()
        return action.astype(np.float32, copy=False)

    def experience(self, views, actions, targets):
        with torch.no_grad():
            return [(model.features(view, torch.as_tensor(view.xy + actions * view.stride
                                                          * view.visible[:, None]))[0], float(target))
                    for model, view, target in zip(self.models, views, targets)]

    def learn(self, batches):
        return self.fit_experiences([self.experience(*batch) for batch in batches])

    def fit_experiences(self, batches):
        losses = []
        for batch in batches:
            self.observations += 1
            for model, (features, target) in zip(self.models, batch):
                model.remember(features, target)
                if self.observations % self.config['fit_interval'] == 0:
                    losses.append(model.fit(self.config, self.fit_rng))
        return float(np.mean(losses)) if losses else 0.

    def policy_state(self):
        return [(model.state_dict(), model.size) for model in self.models]

    def load_policy(self, states):
        for model, (weights, size) in zip(self.models, states):
            model.load_state_dict(weights)
            model.size = size

    def state(self):
        return dict(models=[dict(weights=m.state_dict(), optimizer=m.optimizer.state_dict(),
                                 x=m.replay_x[:m.size].clone(), y=m.replay_y[:m.size].clone(),
                                 size=m.size, next_index=m.next_index) for m in self.models],
                    observations=self.observations, rng=self.fit_rng.get_state())

    def restore(self, state):
        self.observations = state['observations']
        self.fit_rng.set_state(state['rng'])
        for model, saved in zip(self.models, state['models']):
            model.load_state_dict(saved['weights'])
            model.optimizer.load_state_dict(saved['optimizer'])
            model.size, model.next_index = saved['size'], saved['next_index']
            model.replay_x[:model.size].copy_(saved['x'])
            model.replay_y[:model.size].copy_(saved['y'])
