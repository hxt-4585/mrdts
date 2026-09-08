"""Recovered staged PPO sampling, with full physical epochs and bounded storage."""

from dataclasses import replace

import numpy as np
import torch

from experiments.metrics import slot_metrics
from methods.learning.networks.mlp import ValueNetwork
from methods.learning.algorithms.ppo import PPO, PPOSettings, gae_returns
from methods.learning.buffers.rollout import Rollout
from methods.components.flight.ppo import PPOFlight
from methods.components.scheduling.ppo import PPOScheduling
from .method import PPODelayMethod
from .networks import MasterActor, MemberActor
from .observations import MEMBER_FEATURES, MemberPlanning, central_state, master_observations


class Learner:
    def __init__(self, env, hidden=64, settings=None, device='cpu'):
        self.hidden, self.settings = hidden, settings or PPOSettings()
        self.device = torch.device(device)
        obs, _ = master_observations(env)
        global_dim = len(central_state(env))
        self.member = MemberActor(MEMBER_FEATURES, hidden).to(self.device)
        self.master = MasterActor(obs.shape[1], env.member_count * 2, hidden).to(self.device)
        self.member_value = ValueNetwork(global_dim + MEMBER_FEATURES, hidden).to(self.device)
        # Preserve source initialization RNG consumption before replacing the epoch critic.
        ValueNetwork(global_dim + obs.shape[1], hidden)
        self.master_value = ValueNetwork(global_dim + obs.shape[1] + 1, hidden).to(self.device)
        self.member_ppo = PPO(self.member, self.member_value, self.settings)
        self.master_ppo = PPO(self.master, self.master_value, replace(self.settings, learning_rate=1e-4))
        self.flight = PPOFlight(self.master)
        self.scheduling = PPOScheduling(self.member)
        self.dimensions = dict(members=env.member_count, masters=env.master_count,
                               member_features=MEMBER_FEATURES, master_features=obs.shape[1],
                               master_critic_remaining_time=True)

    def save(self, path, metadata):
        from .checkpoint import save_learner
        save_learner(self, path, metadata)

    def load(self, path, optimizers=True, restore_rng=True):
        from .checkpoint import load_learner
        return load_learner(self, path, optimizers, restore_rng)


def as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def pack_rollout(chunks, returns):
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
        actions, log_probs = learner.scheduling.decide(
            torch.as_tensor(obs, device=learner.device), torch.as_tensor(masks, device=learner.device),
            deterministic=deterministic)
        if collect:
            pooled = (obs * masks[:, :, None]).sum(1) / masks.sum(1)[:, None]
            state = np.column_stack([np.broadcast_to(global_state, (len(keys), len(global_state))), pooled]).astype(np.float32)
            values = learner.member_value(torch.as_tensor(state, device=learner.device))
            chunks.append((obs, masks, as_numpy(actions), as_numpy(log_probs), state, as_numpy(values)))
        for key, action in zip(keys, actions.tolist()):
            planning.assign(key, action)
    return planning.placements, chunks


@torch.no_grad()
def choose_master(env, learner, remaining, deterministic):
    obs, masks = master_observations(env)
    global_state = central_state(env)
    state = np.column_stack([np.broadcast_to(global_state, (len(obs), len(global_state))),
                             obs, np.full(len(obs), remaining)]).astype(np.float32)
    actions, latent, log_probs = learner.flight.decide(
        torch.as_tensor(obs, device=learner.device), torch.as_tensor(masks, device=learner.device),
        deterministic=deterministic)
    values = learner.master_value(torch.as_tensor(state, device=learner.device))
    flight = as_numpy(actions.sum(0).reshape(env.member_count, 2))
    return flight, (obs, masks, as_numpy(latent), as_numpy(log_probs), state, as_numpy(values))


def run_epoch(scene, learner, *, stage, steps=500, update_every=8, on_step=None, on_update=None):
    """Consume a fresh scene without resetting it when a Member buffer is flushed."""
    if stage not in ('member', 'master', 'evaluation') or min(steps, update_every) < 1:
        raise ValueError('A valid stage and positive step counts are required')
    method = PPODelayMethod(learner)
    member_chunks, member_returns, master_chunks = [], [], []
    metrics, losses = [], []
    updates = {'member_updates': 0, 'master_updates': 0}
    task_count = 0

    def update(ppo, samples):
        report = ppo.update(samples)
        losses.append(report)
        updates[f'{stage}_updates'] += 1
        if on_update:
            on_update(dict(step=len(metrics), **report))

    for index in range(steps):
        workload = scene.workload(index)
        outcome, row, chunks, master_chunk = method.execute(
            scene, workload, stage=stage, remaining=(steps - index) / steps)
        metrics.append(row)
        tasks = sum(dag.node_num for _, _, dag in workload)
        task_count += tasks
        if on_step:
            on_step(dict(slot_metrics(outcome, 0, index), step=index+1,
                         slot_start=outcome.result.slot_start, slot_end=outcome.result.slot_end,
                         master_calls=1, scheduled_tasks=tasks, **row))
        if stage == 'member':
            member_chunks.extend(chunks)
            # Terminal Monte Carlo return, not repeated instantaneous rewards.
            member_returns.extend([row['reward']] * sum(len(chunk[2]) for chunk in chunks))
            if (index+1) % update_every == 0 or index+1 == steps:
                update(learner.member_ppo, pack_rollout(member_chunks, member_returns))
                member_chunks.clear()
                member_returns.clear()
        elif stage == 'master':
            master_chunks.append(master_chunk)
    if stage == 'master':
        values = np.stack([chunk[-1] for chunk in master_chunks])
        rewards = np.repeat(np.array([row['reward'] for row in metrics])[:,None], len(values[0]), axis=1)
        returns = gae_returns(rewards, values, [False]*(steps-1)+[True])
        samples = pack_rollout(master_chunks, returns.ravel())
        active = samples.masks.any(axis=1)
        samples = Rollout(**{name: getattr(samples,name)[active] for name in samples.__dataclass_fields__})
        update(learner.master_ppo, samples)
    result = {name: float(np.mean([row[name] for row in metrics])) for name in metrics[0]}
    result.update(steps=steps, master_calls=steps, scheduled_tasks=task_count,
                  epoch_return=float(sum(row['reward'] for row in metrics)), mean_reward=result['reward'], **updates)
    if losses:
        result.update({name: float(np.mean([row[name] for row in losses])) for name in losses[0]})
    return result
