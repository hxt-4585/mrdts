"""Independent regional delay surrogate and one-hop distributed flight planner."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RegionView:
    positions: np.ndarray
    visible: np.ndarray
    owned: np.ndarray
    users: np.ndarray
    neighbors: tuple
    step: float


def observe(env):
    count = env.members.member_uav_count
    region_ids = env.members.region_ids[:count]
    owners = {int(r): i for i, r in enumerate(env.masters.region_ids)}
    scale = env.region.config.side_length
    distance = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s / scale
    views = []
    for master, region in enumerate(env.masters.region_ids):
        adjacent = env.masters.neighbor_region_ids[int(region)]
        views.append(RegionView(
            env.observe_master(master).reshape(count, 3)[:, :2].astype(np.float32),
            np.isin(region_ids, [region, *adjacent]), region_ids == region,
            (env.users.positions[env.users.region_ids == region, :2] / scale).astype(np.float32),
            tuple(owners[r] for r in adjacent), distance))
    return views


class RegionalDelay(nn.Module):
    def __init__(self, view, config):
        super().__init__()
        count, users = len(view.positions), len(view.users)
        self.register_buffer('user_xy', torch.tensor(view.users))
        width = 4 * count + count * users + users
        hidden = config['hidden']
        self.regressor = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['learning_rate'])
        self.inputs = torch.empty(config['replay_size'], width)
        self.labels = torch.empty(config['replay_size'])
        self.samples = self.cursor = 0

    def encode(self, view, destination):
        xy = destination.reshape(-1, len(view.positions), 2)
        visible = torch.as_tensor(view.visible, dtype=xy.dtype)
        owned = torch.as_tensor(view.owned, dtype=xy.dtype)
        squared = (xy[:, :, None] - self.user_xy).square().sum(-1)
        squared = squared * visible[None, :, None] + 2 * (1 - visible[None, :, None])
        nearest = -.02 * torch.logsumexp(-squared / .02, dim=1)
        return torch.cat(((xy * visible[None, :, None]).flatten(1),
                          visible.expand(len(xy), -1), owned.expand(len(xy), -1),
                          squared.flatten(1), nearest), dim=1)

    def delay(self, view, destination):
        return F.softplus(self.regressor(self.encode(view, destination))).flatten()

    def append(self, features, target):
        self.inputs[self.cursor].copy_(torch.as_tensor(features))
        self.labels[self.cursor] = target
        self.cursor = (self.cursor + 1) % len(self.labels)
        self.samples = min(self.samples + 1, len(self.labels))

    def fit(self, config, generator):
        losses = []
        for _ in range(config['fit_steps']):
            ix = torch.randint(self.samples, (min(config['batch_size'], self.samples),), generator=generator)
            estimate = F.softplus(self.regressor(self.inputs[ix])).flatten()
            loss = F.mse_loss(torch.log1p(estimate), torch.log1p(self.labels[ix]))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach()))
        return sum(losses) / len(losses)

    def snapshot(self):
        return dict(model=self.state_dict(), optimizer=self.optimizer.state_dict(),
                    samples=self.samples, cursor=self.cursor,
                    inputs=self.inputs[:self.samples].clone(), labels=self.labels[:self.samples].clone())

    def restore(self, state):
        self.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.samples, self.cursor = state['samples'], state['cursor']
        self.inputs[:self.samples].copy_(state['inputs'])
        self.labels[:self.samples].copy_(state['labels'])


def coverage(view):
    ids = np.flatnonzero(view.owned)
    action = np.zeros_like(view.positions)
    if len(ids):
        centers = view.positions[ids].copy()
        for _ in range(6):
            cluster = np.sum((view.users[:, None] - centers[None]) ** 2, axis=-1).argmin(axis=1)
            for i in range(len(ids)):
                assigned = view.users[cluster == i]
                if len(assigned):
                    centers[i] = assigned.mean(axis=0)
        delta = np.clip(centers - view.positions[ids], -view.step, view.step)
        action[ids] = np.clip(delta, -view.positions[ids], 1 - view.positions[ids]) / view.step
    return action


def optimize(views, models, seed, config):
    plans, limits = [], []
    margin = 4 * np.finfo(np.float32).eps
    for view in views:
        ids = np.flatnonzero(view.owned)
        low = np.maximum(-view.step, margin - view.positions[ids])
        high = np.minimum(view.step, 1 - margin - view.positions[ids])
        plans.append(np.clip(seed[ids] * view.step, low, high))
        limits.append((ids, low, high))
    for iteration in range(config['rounds']):
        derivatives = [np.zeros_like(plan) for plan in plans]
        for owner, (view, model) in enumerate(zip(views, models)):
            destination = view.positions.copy()
            participants = (owner, *view.neighbors)
            for sender in participants:
                ids = limits[sender][0]
                destination[ids] += plans[sender] * view.visible[ids, None]
            point = torch.tensor(destination, requires_grad=True)
            cost = model.delay(view, point).sum() * len(view.users)
            gradient = torch.autograd.grad(cost, point)[0].numpy()
            for receiver in participants:
                ids = limits[receiver][0]
                derivatives[receiver] += gradient[ids] * view.visible[ids, None]
        for i, gradient in enumerate(derivatives):
            length = max(1., float(np.linalg.norm(gradient)))
            rate = config['gradient_rate'] / math.sqrt(iteration + 1)
            plans[i] = np.clip(plans[i] - rate * gradient / length, limits[i][1], limits[i][2])
    result = np.zeros_like(seed)
    for view, plan, (ids, _, _) in zip(views, plans, limits):
        result[ids] = plan / view.step
    return result


class Controller:
    def __init__(self, views, config):
        self.config = config
        # Identical high-level initialization regardless of the low-level architecture.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'] + 100)
            self.models = [RegionalDelay(view, config) for view in views]
        self.generator = torch.Generator().manual_seed(config['seed'] + 101)
        self.observations = 0
        self.reset(0)

    def reset(self, epoch):
        self.rng = [np.random.default_rng(np.random.SeedSequence([self.config['seed'], epoch, owner]))
                    for owner in range(len(self.models))]
        self.slot = 0
        self.previous = None
        self.goals = None
        self.goal_owners = None

    def actions(self, views, exploration):
        seed = np.sum([coverage(view) for view in views], axis=0)
        if self.previous is not None:
            seed = (seed + self.previous) * .5
        actions = (optimize(views, self.models, seed, self.config)
                   if min(m.samples for m in self.models) >= self.config['warmup_samples'] else seed)
        if exploration:
            if self.goals is None or self.slot % self.config['exploration_hold'] == 0:
                self.goals = [np.zeros_like(actions) for _ in views]
                self.goal_owners = [np.zeros(len(actions), dtype=bool) for _ in views]
                self.exploring = [rng.random() < exploration for rng in self.rng]
            for i, view in enumerate(views):
                arrivals = np.flatnonzero(view.owned & ~self.goal_owners[i])
                goals = view.users[self.rng[i].integers(len(view.users), size=len(arrivals))]
                self.goals[i][arrivals] = np.clip(
                    goals + self.rng[i].normal(0., .12, goals.shape), .001, .999)
                self.goal_owners[i] = view.owned.copy()
                if self.exploring[i]:
                    ids = np.flatnonzero(view.owned)
                    actions[ids] = np.clip((self.goals[i][ids] - view.positions[ids]) / view.step, -1., 1.)
        self.previous = actions.copy()
        self.slot += 1
        return actions.astype(np.float32, copy=False)

    def experience(self, views, actions, regional_delays):
        with torch.no_grad():
            return [(model.encode(view, torch.as_tensor(
                        view.positions + actions * view.step * view.visible[:, None]))[0].numpy(),
                     float(delay / len(view.users)))
                    for model, view, delay in zip(self.models, views, regional_delays)]

    def feedback(self, experiences):
        losses = []
        for experience in experiences:
            self.observations += 1
            for model, (features, target) in zip(self.models, experience):
                model.append(features, target)
                if self.observations % self.config['fit_interval'] == 0:
                    losses.append(model.fit(self.config, self.generator))
        return float(np.mean(losses)) if losses else 0.

    def policy(self):
        return [(model.state_dict(), model.samples) for model in self.models]

    def load_policy(self, state):
        for model, (weights, samples) in zip(self.models, state):
            model.load_state_dict(weights)
            model.samples = samples

