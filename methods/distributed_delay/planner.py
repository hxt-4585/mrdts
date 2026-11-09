"""One-hop messages and fixed-round, synchronous projected gradient planning.

Frames contain no simulator or current/future DAG. Communication is explicitly
restricted to action proposals and partial gradients on graph edges.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Frame:
    xy: np.ndarray
    visible: np.ndarray
    owned: np.ndarray
    users: np.ndarray
    neighbors: tuple
    step: float


def frames_from_env(env):
    """Telemetry boundary: one-hop UAV positions and own-region user locations."""
    count = env.members.member_uav_count
    side = env.region.config.side_length
    regions = env.members.region_ids[:count]
    ids = {int(region): index for index, region in enumerate(env.masters.region_ids)}
    step = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s / side
    frames = []
    for owner, region in enumerate(env.masters.region_ids):
        neighbors = env.masters.neighbor_region_ids[int(region)]
        xy = env.observe_master(owner).reshape(count, 3)[:, :2].astype(np.float32)
        frames.append(Frame(xy, np.isin(regions, (region, *neighbors)), regions == region,
                            (env.users.positions[env.users.region_ids == region, :2] / side).astype(np.float32),
                            tuple(ids[neighbor] for neighbor in neighbors), step))
    return frames


class LocalProxy(nn.Module):
    """Smooth regional mean-delay regression; replay contains local samples only."""
    def __init__(self, members, users, *, hidden=64, learning_rate=.001, replay_size=4096):
        super().__init__()
        self.register_buffer('users', torch.as_tensor(users, dtype=torch.float32))
        width = 4 * members + members * len(users) + len(users)
        self.network = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.replay_size = replay_size
        self.samples = 0
        self.write_index = 0
        self.x = torch.empty((replay_size, width))
        self.y = torch.empty(replay_size)

    def features(self, endpoint, visible, owned):
        endpoint = endpoint.reshape(-1, len(visible), 2)
        visible = torch.as_tensor(visible, dtype=torch.float32)
        owned = torch.as_tensor(owned, dtype=torch.float32)
        masked = endpoint * visible[None, :, None]
        distances = (endpoint[:, :, None, :] - self.users[None, None]).square().sum(-1)
        distances = distances * visible[None, :, None] + (1. - visible[None, :, None]) * 2.
        # Smooth nearest-distance summary supplies useful geometry without hard argmin.
        nearest = -.02 * torch.logsumexp(-distances / .02, dim=1)
        batch = endpoint.shape[0]
        return torch.cat((masked.flatten(1), visible.expand(batch, -1), owned.expand(batch, -1),
                          distances.flatten(1), nearest), dim=1)

    def value(self, frame, endpoint):
        return F.softplus(self.network(self.features(endpoint, frame.visible, frame.owned))).squeeze(-1)

    def remember(self, frame, endpoint, delay):
        with torch.no_grad():
            self.x[self.write_index] = self.features(torch.as_tensor(endpoint), frame.visible, frame.owned)[0]
            self.y[self.write_index] = delay
        self.write_index = (self.write_index + 1) % self.replay_size
        self.samples = min(self.samples + 1, self.replay_size)

    def fit(self, *, steps, batch_size):
        total = 0.
        for _ in range(steps):
            indices = torch.randint(self.samples, (min(batch_size, self.samples),))
            prediction = F.softplus(self.network(self.x[indices])).flatten()
            loss = F.mse_loss(torch.log1p(prediction), torch.log1p(self.y[indices]))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
        return total / steps

    def training_state(self):
        return dict(model=self.state_dict(), optimizer=self.optimizer.state_dict(), samples=self.samples,
                    write_index=self.write_index, x=self.x[:self.samples].clone(), y=self.y[:self.samples].clone())

    def restore(self, saved):
        self.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.samples, self.write_index = saved['samples'], saved['write_index']
        self.x[:self.samples] = saved['x']
        self.y[:self.samples] = saved['y']


def coordinate(frames, models, initial, *, rounds, rate):
    """Each model sees its own frame and proposals from direct neighbors only.

    Each gradient reply contains only derivatives with respect to its recipient's
    owned variables. All models evaluate the same round snapshot before updates.
    """
    plans = []
    for frame in frames:
        ids = np.flatnonzero(frame.owned)
        # Keep a few float32 ULPs inside the physical boundary: normalized XY
        # is converted back to metres by the environment's float32 integrator.
        margin = 4 * np.finfo(np.float32).eps
        lower = np.maximum(-frame.step, margin - frame.xy[ids])
        upper = np.minimum(frame.step, 1. - margin - frame.xy[ids])
        plans.append([ids, np.clip(initial[ids] * frame.step, lower, upper), lower, upper])
    messages = payload = evaluations = 0
    for iteration in range(rounds):
        inbox = [np.zeros_like(plan[1]) for plan in plans]
        for receiver, (frame, model) in enumerate(zip(frames, models)):
            endpoint = frame.xy.copy()
            for sender in (receiver, *frame.neighbors):
                ids, displacement = plans[sender][:2]
                endpoint[ids] += displacement * frame.visible[ids, None]
                if sender != receiver:
                    payload += displacement.size
            point = torch.tensor(endpoint, requires_grad=True)
            objective = model.value(frame, point).sum() * len(frame.users)
            gradient = torch.autograd.grad(objective, point)[0].detach().numpy()
            evaluations += 1
            for recipient in (receiver, *frame.neighbors):
                ids = plans[recipient][0]
                inbox[recipient] += gradient[ids] * frame.visible[ids, None]
                if recipient != receiver:
                    messages += 1
                    payload += 2 * len(ids)
        for plan, gradient in zip(plans, inbox):
            # Normalize only the update direction; the objective remains pure delay.
            scale = max(1., float(np.linalg.norm(gradient)))
            update = rate / (iteration + 1) ** .5 * gradient / scale
            plan[1] = np.clip(plan[1] - update, plan[2], plan[3])
    actions = np.zeros_like(initial, dtype=np.float32)
    for frame, (ids, displacement, _, _) in zip(frames, plans):
        actions[ids] = displacement / frame.step
    return actions, dict(gradient_messages=messages, action_messages=messages,
                         payload_scalars=payload, proxy_evaluations=evaluations)


def coverage_actions(frame):
    """Local Lloyd descent initializes the optimizer without CEM supervision."""
    ids = np.flatnonzero(frame.owned)
    result = np.zeros_like(frame.xy)
    centers = frame.xy[ids].copy()
    if len(ids):
        for _ in range(6):
            association = ((frame.users[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
            for index in range(len(ids)):
                assigned = frame.users[association == index]
                if len(assigned):
                    centers[index] = assigned.mean(0)
        displacement = np.clip(centers - frame.xy[ids], -frame.step, frame.step)
        displacement = np.clip(displacement, -frame.xy[ids], 1. - frame.xy[ids])
        result[ids] = displacement / frame.step
    return result


class DistributedController:
    def __init__(self, frames, config):
        self.config = config
        self.models = [LocalProxy(len(frame.xy), frame.users, hidden=config['hidden'],
                       learning_rate=config['learning_rate'], replay_size=config['replay_size']) for frame in frames]
        self.reset(0)

    def reset(self, epoch):
        self.rngs = [np.random.default_rng(np.random.SeedSequence([self.config['seed'], epoch, owner]))
                     for owner in range(len(self.models))]
        self.slot = 0
        self.previous = None
        self.goals = None
        self.goal_owners = None

    def actions(self, frames, *, exploration):
        seed = sum((coverage_actions(frame) for frame in frames), np.zeros_like(frames[0].xy))
        if self.previous is not None:
            seed = .5 * seed + .5 * self.previous
        if min(model.samples for model in self.models) >= self.config['warmup_samples']:
            actions, stats = coordinate(frames, self.models, seed, rounds=self.config['rounds'],
                                        rate=self.config['gradient_rate'])
        else:
            actions = seed
            stats = dict(gradient_messages=0, action_messages=0, payload_scalars=0, proxy_evaluations=0)
        if exploration:
            if self.goals is None or self.slot % self.config['exploration_hold'] == 0:
                self.goals = [np.zeros_like(actions) for _ in frames]
                self.goal_owners = [np.zeros(len(actions), bool) for _ in frames]
                self.exploring = []
                for owner, frame in enumerate(frames):
                    rng = self.rngs[owner]
                    self.exploring.append(rng.random() < exploration)
            for owner, frame in enumerate(frames):
                # Ownership handoffs do not transfer another Master's private
                # exploration goal. New arrivals receive a locally sampled goal.
                rng = self.rngs[owner]
                arrivals = np.flatnonzero(frame.owned & ~self.goal_owners[owner])
                goals = frame.users[rng.integers(len(frame.users), size=len(arrivals))]
                self.goals[owner][arrivals] = np.clip(goals + rng.normal(0., .12, goals.shape), .001, .999)
                self.goal_owners[owner] = frame.owned.copy()
                if self.exploring[owner]:
                    ids = np.flatnonzero(frame.owned)
                    actions[ids] = np.clip((self.goals[owner][ids] - frame.xy[ids]) / frame.step, -1., 1.)
        # Broadcast final decisions after projection/exploration. These edge
        # messages also make feedback endpoints reconstructible locally.
        stats['action_messages'] += sum(len(frame.neighbors) for frame in frames)
        stats['payload_scalars'] += sum(2 * int(frame.owned.sum()) * len(frame.neighbors) for frame in frames)
        self.previous = actions.copy()
        self.slot += 1
        return actions, stats

    def feedback(self, frames, actions, regional_delays):
        losses = []
        for owner, (frame, model) in enumerate(zip(frames, self.models)):
            endpoint = frame.xy + actions * frame.step * frame.visible[:, None]
            model.remember(frame, endpoint, float(regional_delays[owner] / len(frame.users)))
            if self.slot % self.config['fit_interval'] == 0:
                losses.append(model.fit(steps=self.config['fit_steps'], batch_size=self.config['batch_size']))
        return float(np.mean(losses)) if losses else 0.
