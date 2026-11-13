"""Regional delay regressors and projected one-hop flight planning."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Scene:
    xy: np.ndarray
    visible: np.ndarray  # [region, member]
    owned: np.ndarray

    def encode(self, positions):
        visible = torch.as_tensor(self.visible, dtype=positions.dtype)
        owned = torch.as_tensor(self.owned, dtype=positions.dtype)
        return torch.cat(((positions * visible.unsqueeze(-1)).flatten(1), visible, owned), dim=1)


class RegionalDelays:
    def __init__(self, members, regions, config):
        self.config = config
        width = 4 * members
        hidden = config['hidden']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'] + 100)
            self.networks = nn.ModuleList(
                nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                              nn.SiLU(), nn.Linear(hidden, 1), nn.Softplus()) for _ in range(regions))
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
                           for net in self.networks]
        capacity = config['replay_size']
        self.inputs = torch.empty(capacity, regions, width)
        self.labels = torch.empty(capacity, regions)
        self.seen = 0
        self.rng = torch.Generator().manual_seed(config['seed'] + 101)

    def update(self, experiences):
        cfg = self.config
        capacity = len(self.inputs)
        losses = []
        for inputs, labels in experiences:
            index = self.seen % capacity
            self.inputs[index].copy_(inputs)
            self.labels[index].copy_(torch.as_tensor(labels))
            self.seen += 1
            if self.seen % cfg['fit_interval']:
                continue
            size = min(self.seen, capacity)
            for region, (net, optimizer) in enumerate(zip(self.networks, self.optimizers)):
                for _ in range(cfg['fit_steps']):
                    batch = torch.randint(size, (min(cfg['batch_size'], size),), generator=self.rng)
                    predicted = net(self.inputs[batch, region]).flatten()
                    loss = F.mse_loss(predicted.log1p(), self.labels[batch, region].log1p())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach().item())
        return float(np.mean(losses)) if losses else 0.

    def checkpoint(self):
        size = min(self.seen, len(self.labels))
        return dict(weights=self.networks.state_dict(), seen=self.seen,
                    inputs=self.inputs[:size].clone(), labels=self.labels[:size].clone(),
                    optimizers=[opt.state_dict() for opt in self.optimizers], rng=self.rng.get_state())

    def restore(self, state, training):
        self.networks.load_state_dict(state['weights'])
        self.seen = state['seen']
        if training:
            size = len(state['labels'])
            self.inputs[:size].copy_(state['inputs'])
            self.labels[:size].copy_(state['labels'])
            self.rng.set_state(state['rng'])
            for opt, saved in zip(self.optimizers, state['optimizers']):
                opt.load_state_dict(saved)


class FlightController:
    def __init__(self, env, delays, epoch):
        self.delays = delays
        self.cfg = delays.config
        self.scale = env.region.config.side_length
        self.stride = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s / self.scale
        self.regions = env.masters.region_ids
        self.neighborhoods = [(r, *env.masters.neighbor_region_ids[r]) for r in self.regions]
        self.users = [(env.users.positions[env.users.region_ids == r, :2] / self.scale).astype(np.float32)
                      for r in self.regions]
        self.random = [np.random.default_rng([self.cfg['seed'], epoch, r]) for r in range(len(self.regions))]
        self.exploring = np.zeros(len(self.regions), bool)
        self.goals = np.zeros((env.members.member_uav_count, 2), np.float32)
        self.goal_owner = np.full(len(self.goals), -1)
        self.last = None
        self.tick = 0

    def observe(self, env):
        count = len(self.goals)
        regions = env.members.region_ids[:count]
        return Scene((env.members.positions[:count, :2] / self.scale).astype(np.float32),
                     np.stack([np.isin(regions, neighborhood) for neighborhood in self.neighborhoods]),
                     regions == self.regions[:, None])

    def coverage_action(self, xy, groups):
        action = np.zeros_like(xy)
        for ids, users in zip(groups, self.users):
            if not len(ids):
                continue
            centers = xy[ids].copy()
            for _ in range(6):
                nearest = ((users[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
                for i in range(len(centers)):
                    assigned = users[nearest == i]
                    if len(assigned):
                        centers[i] = assigned.mean(axis=0)
            action[ids] = np.clip((centers - xy[ids]) / self.stride, -1., 1.)
        return action

    def refine(self, scene, groups, action):
        margin = 4 * np.finfo(np.float32).eps
        lower = np.maximum(-self.stride, margin - scene.xy)
        upper = np.minimum(self.stride, 1 - margin - scene.xy)
        shift = np.clip(action * self.stride, lower, upper)
        for i in range(self.cfg['rounds']):
            destination = torch.tensor(scene.xy + shift, requires_grad=True)
            features = scene.encode(destination)
            # Visibility masks route each region's gradient only to its one-hop members.
            delay = sum(net(features[r]).sum() for r, net in enumerate(self.delays.networks))
            gradient = torch.autograd.grad(delay, destination)[0].numpy()
            step = self.cfg['gradient_rate'] / math.sqrt(i + 1)
            for ids in groups:
                local = gradient[ids]
                shift[ids] -= step * local / max(1., float(np.linalg.norm(local)))
            np.clip(shift, lower, upper, out=shift)
        return shift / self.stride

    def act(self, scene, exploration):
        groups = [np.flatnonzero(mask) for mask in scene.owned]
        action = self.coverage_action(scene.xy, groups)
        if self.last is not None:
            action = (action + self.last) * .5
        if min(self.delays.seen, len(self.delays.labels)) >= self.cfg['warmup_samples']:
            action = self.refine(scene, groups, action)
        if exploration:
            if self.tick % self.cfg['exploration_hold'] == 0:
                self.goal_owner.fill(-1)
                self.exploring = [rng.random() < exploration for rng in self.random]
            for r, ids in enumerate(groups):
                rng = self.random[r]
                arrivals = ids[self.goal_owner[ids] != r]
                users = self.users[r]
                targets = users[rng.integers(len(users), size=len(arrivals))]
                self.goals[arrivals] = np.clip(targets + rng.normal(0., .12, targets.shape), .001, .999)
                self.goal_owner[ids] = r
                if self.exploring[r]:
                    action[ids] = np.clip((self.goals[ids] - scene.xy[ids]) / self.stride, -1., 1.)
        self.tick += 1
        self.last = action
        return action
