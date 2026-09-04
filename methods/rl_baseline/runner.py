"""Rollouts, staged learners and checkpoint serialization."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .networks import MasterActor, MemberActor, ValueNetwork
from .observations import MEMBER_FEATURES, MemberPlanning, central_state, master_observations
from .ppo import PPO, PPOSettings, Rollout, gae_returns


class Learner:
    def __init__(self, env, hidden=64, settings=None):
        self.hidden = hidden
        self.settings = settings or PPOSettings()
        self.member = MemberActor(MEMBER_FEATURES, hidden)
        obs, _ = master_observations(env)
        global_dim = len(central_state(env))
        self.master = MasterActor(obs.shape[1], env.member_count * 2, hidden)
        self.member_value = ValueNetwork(global_dim + MEMBER_FEATURES, hidden)
        self.master_value = ValueNetwork(global_dim + obs.shape[1], hidden)
        self.member_ppo = PPO(self.member, self.member_value, self.settings)
        self.master_ppo = PPO(self.master, self.master_value,
                              replace(self.settings, learning_rate=1e-4))
        self.dimensions = dict(members=env.member_count, masters=env.master_count,
                               member_features=MEMBER_FEATURES, master_features=obs.shape[1])

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(format_version=2, hidden=self.hidden, settings=asdict(self.settings),
                     dimensions=self.dimensions, metadata=metadata, rng=torch.get_rng_state())
        for name in ('member', 'master', 'member_value', 'master_value'):
            state[name] = getattr(self, name).state_dict()
        for name in ('member_ppo', 'master_ppo'):
            ppo = getattr(self, name)
            state[name] = dict(actor=ppo.actor_optimizer.state_dict(),
                               critic=ppo.critic_optimizer.state_dict())
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path, optimizers=True, restore_rng=True):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('format_version') != 2 or state['dimensions'] != self.dimensions:
            raise ValueError('Checkpoint schema or environment dimensions do not match')
        for name in ('member', 'master', 'member_value', 'master_value'):
            getattr(self, name).load_state_dict(state[name])
        if optimizers:
            for name in ('member_ppo', 'master_ppo'):
                ppo = getattr(self, name)
                ppo.actor_optimizer.load_state_dict(state[name]['actor'])
                ppo.critic_optimizer.load_state_dict(state[name]['critic'])
        if restore_rng:
            torch.set_rng_state(state['rng'])
        return state['metadata']


def _rollout(chunks, returns):
    columns = [np.concatenate([chunk[i] for chunk in chunks]) for i in range(6)]
    return Rollout(*columns, np.asarray(returns, dtype=np.float32))


@torch.no_grad()
def plan_members(env, batch, learner, deterministic=False, collect=False):
    planning = MemberPlanning(env, batch)
    orders = list(batch.plan.member_orders.values())
    global_state = central_state(env, batch) if collect else None
    chunks = []
    for step in range(max(map(len, orders), default=0)):
        keys = [order[step] for order in orders if step < len(order)]
        features = [planning.observe(key) for key in keys]
        obs = np.stack([item[0] for item in features])
        masks = np.stack([item[1] for item in features])
        tensor_obs, tensor_masks = torch.from_numpy(obs), torch.from_numpy(masks)
        dist = learner.member.distribution(tensor_obs, tensor_masks)
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        if collect:
            pooled = (obs * masks[:, :, None]).sum(1) / masks.sum(1)[:, None]
            state = np.column_stack([np.broadcast_to(global_state, (len(keys), len(global_state))), pooled]).astype(np.float32)
            values = learner.member_value(torch.from_numpy(state)).numpy()
            chunks.append((obs, masks, actions.numpy(), dist.log_prob(actions).numpy(), state, values))
        for key, action in zip(keys, actions.tolist()):
            planning.assign(key, action)
    return planning.placements, chunks


@torch.no_grad()
def choose_master(env, learner, deterministic=False):
    obs, masks = master_observations(env)
    global_state = central_state(env)
    state = np.column_stack([np.broadcast_to(global_state, (len(obs), len(global_state))), obs]).astype(np.float32)
    observations = torch.from_numpy(obs)
    mask = torch.from_numpy(masks)
    dist = learner.master.distribution(observations)
    latent = dist.mean if deterministic else dist.sample()
    log_probs, _ = learner.master.evaluate(observations, mask, latent)
    values = learner.master_value(torch.from_numpy(state))
    # Disjoint ownership: each physical UAV receives exactly one Master's action.
    joint_actions = learner.master.actions(latent, mask).sum(0).reshape(env.member_count, 2).numpy()
    return joint_actions, (obs, masks, latent.numpy(), log_probs.numpy(), state, values.numpy())


def collect_member(env, learner, slots):
    env.reset()
    chunks, returns, metrics = [], [], []
    for _ in range(slots):
        batch = env.begin()
        actions, step_chunks = plan_members(env, batch, learner, collect=True)
        result = env.finish(batch, actions)
        # One placement episode per slot; all interior rewards are zero. This is
        # the terminal return, NOT a reward repeatedly emitted for every task.
        returns.extend([result['reward']] * sum(len(chunk[2]) for chunk in step_chunks))
        chunks.extend(step_chunks)
        metrics.append(result)
    return _rollout(chunks, returns), metrics


def collect_master(env, learner, slots):
    env.reset()
    chunks, metrics = [], []
    for _ in range(slots):
        actions, chunk = choose_master(env, learner)
        batch = env.begin(actions)
        placements, _ = plan_members(env, batch, learner, deterministic=True)
        metrics.append(env.finish(batch, placements))
        chunks.append(chunk)
    values = np.stack([chunk[-1] for chunk in chunks])
    rewards = np.repeat(np.array([m['reward'] for m in metrics])[:, None], env.master_count, axis=1)
    returns = gae_returns(rewards, values, [False] * (slots - 1) + [True])
    # Empty regions have no controllable action and must not contribute fake PPO samples.
    rollout = _rollout(chunks, returns.ravel())
    active = rollout.masks.any(axis=1)
    return Rollout(**{name: getattr(rollout, name)[active] for name in rollout.__dataclass_fields__}), metrics


def evaluate(env, learner, slots=4, flight=False, baseline=None):
    env.reset()
    rows = []
    for _ in range(slots):
        actions = choose_master(env, learner, deterministic=True)[0] if flight else None
        batch = env.begin(actions)
        if baseline == 'ground':
            placements = {key: 0 for key in batch.plan.order}
        elif baseline == 'owner':
            placements = {key: key.owner_member_id + 1 for key in batch.plan.order}
        elif baseline is None:
            placements, _ = plan_members(env, batch, learner, deterministic=True)
        else:
            raise ValueError('Unknown evaluation baseline')
        rows.append(env.finish(batch, placements))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
