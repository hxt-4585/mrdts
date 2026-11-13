"""Independent regional delay learning and one-hop projected flight descent."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


@dataclass
class FlightScene:
    positions: np.ndarray
    visible: np.ndarray
    ownership: np.ndarray

    def features(self, positions):
        visibility = torch.as_tensor(self.visible, dtype=positions.dtype)
        ownership = torch.as_tensor(self.ownership, dtype=positions.dtype)
        coordinates = positions.unsqueeze(0) * visibility.unsqueeze(2)
        return torch.cat([coordinates.flatten(1), visibility, ownership], dim=1)


class FlightPolicy:
    """Regional regressors predict contributions to the global mean DAG delay.

    Movement gradients use only own/adjacent-region visibility. Exploration
    holds user-directed goals for several slots; evaluation disables it.
    """
    def __init__(self, env, config):
        self.config = config
        self.member_count = env.members.member_uav_count
        self.region_ids = tuple(int(value) for value in env.masters.region_ids)
        self.side = env.region.config.side_length
        self.stride = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s / self.side
        self.visibility_regions = [(region, *env.masters.neighbor_region_ids[region])
                                   for region in self.region_ids]
        self.user_points = [(env.users.positions[env.users.region_ids == region, :2] / self.side)
                            .astype(np.float32) for region in self.region_ids]
        width, hidden = 4 * self.member_count, config['hidden']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'] + 100)
            self.networks = nn.ModuleList()
            for _ in self.region_ids:
                self.networks.append(nn.Sequential(
                    nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                    nn.SiLU(), nn.Linear(hidden, 1), nn.Softplus()))
        self.optimizers = [torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
                           for network in self.networks]
        self.replay_features = torch.empty(config['replay_size'], len(self.region_ids), width)
        self.replay_targets = torch.empty(config['replay_size'], len(self.region_ids))
        self.samples = 0
        self.batches = torch.Generator().manual_seed(config['seed'] + 101)

    def reset(self, epoch, sampler=0):
        self.rngs = [np.random.default_rng([self.config['seed'], epoch, region, sampler])
                     for region in self.region_ids]
        self.steps = 0
        self.previous_action = None
        self.exploring = np.zeros(len(self.region_ids), dtype=bool)
        self.goal_regions = np.full(self.member_count, -1)
        self.goals = np.zeros((self.member_count, 2), dtype=np.float32)

    def observe(self, env):
        regions = env.members.region_ids[:self.member_count]
        positions = (env.members.positions[:self.member_count, :2] / self.side).astype(np.float32)
        return FlightScene(positions,
                           np.array([np.isin(regions, visible) for visible in self.visibility_regions]),
                           np.array([regions == region for region in self.region_ids]))

    def act(self, scene, exploration):
        cfg = self.config
        groups = [np.flatnonzero(owned) for owned in scene.ownership]
        action = np.zeros_like(scene.positions)
        # Coverage initialization, with persistent user-directed exploration.
        for group, users in zip(groups, self.user_points):
            if group.size == 0:
                continue
            centers = scene.positions[group].copy()
            for _ in range(6):
                assignments = np.linalg.norm(users[:, None, :] - centers[None, :, :], axis=2).argmin(1)
                for index in range(group.size):
                    selected = users[assignments == index]
                    if selected.size:
                        centers[index] = selected.mean(0)
            action[group] = np.clip((centers - scene.positions[group]) / self.stride, -1., 1.)
        if self.previous_action is not None:
            action = .5 * (action + self.previous_action)

        margin = 4 * np.finfo(np.float32).eps
        lower = np.maximum(-self.stride, margin - scene.positions)
        upper = np.minimum(self.stride, 1. - margin - scene.positions)
        displacement = np.clip(action * self.stride, lower, upper)
        if min(self.samples, len(self.replay_targets)) >= cfg['warmup_samples']:
            for round_index in range(cfg['rounds']):
                destination = torch.tensor(scene.positions + displacement, requires_grad=True)
                features = scene.features(destination)
                objective = torch.stack([network(features[index]).sum()
                                         for index, network in enumerate(self.networks)]).sum()
                derivative = torch.autograd.grad(objective, destination)[0].numpy()
                rate = cfg['gradient_rate'] / math.sqrt(round_index + 1)
                for group in groups:
                    gradient = derivative[group]
                    displacement[group] -= rate * gradient / max(float(np.linalg.norm(gradient)), 1.)
                np.clip(displacement, lower, upper, out=displacement)
        action = displacement / self.stride

        if exploration > 0.:
            if self.steps % cfg['exploration_hold'] == 0:
                self.goal_regions.fill(-1)
                self.exploring = np.array([rng.random() < exploration for rng in self.rngs])
            for region, (group, users, rng) in enumerate(zip(groups, self.user_points, self.rngs)):
                arrivals = group[self.goal_regions[group] != region]
                targets = users[rng.integers(users.shape[0], size=arrivals.size)]
                self.goals[arrivals] = np.clip(rng.normal(targets, .12), .001, .999)
                self.goal_regions[group] = region
                if self.exploring[region]:
                    action[group] = np.clip((self.goals[group] - scene.positions[group]) / self.stride, -1., 1.)
        self.steps += 1
        self.previous_action = action.copy()
        return action

    def learn(self, experiences):
        cfg = self.config
        losses = []
        for features, targets in experiences:
            position = self.samples % cfg['replay_size']
            self.replay_features[position] = features
            self.replay_targets[position] = torch.as_tensor(targets)
            self.samples += 1
            if self.samples % cfg['fit_interval'] != 0:
                continue
            size = min(self.samples, cfg['replay_size'])
            for region, network in enumerate(self.networks):
                optimizer = self.optimizers[region]
                for _ in range(cfg['fit_steps']):
                    indices = torch.randint(size, (min(size, cfg['batch_size']),), generator=self.batches)
                    predictions = network(self.replay_features[indices, region]).squeeze(1)
                    labels = self.replay_targets[indices, region]
                    loss = torch.mean((torch.log1p(predictions) - torch.log1p(labels)) ** 2)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
        return math.fsum(losses) / len(losses) if losses else 0.

    def state(self):
        size = min(self.samples, self.config['replay_size'])
        return dict(weights=self.networks.state_dict(), samples=self.samples,
                    features=self.replay_features[:size].clone(), targets=self.replay_targets[:size].clone(),
                    optimizers=[optimizer.state_dict() for optimizer in self.optimizers],
                    batch_rng=self.batches.get_state())

    def restore(self, state):
        self.networks.load_state_dict(state['weights'])
        self.samples = state['samples']
        size = len(state['targets'])
        self.replay_features[:size] = state['features']
        self.replay_targets[:size] = state['targets']
        self.batches.set_state(state['batch_rng'])
        for optimizer, saved in zip(self.optimizers, state['optimizers']):
            optimizer.load_state_dict(saved)
