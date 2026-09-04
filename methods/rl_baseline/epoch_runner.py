"""Complete Master-plus-Member epochs, with bounded Member rollout memory."""

from dataclasses import replace

import numpy as np
import torch

from .networks import ValueNetwork
from .observations import central_state, master_observations
from .ppo import PPO, Rollout, gae_returns
from .runner import Learner, _rollout, plan_members


def master_critic_input(env, remaining):
    observations, _ = master_observations(env)
    state = central_state(env)
    return np.column_stack([np.broadcast_to(state, (len(observations), len(state))),
                            observations, np.full(len(observations), remaining)]).astype(np.float32)


class EpochLearner(Learner):
    """Reuse actors and serialization; mark the extended critic schema explicitly."""

    def __init__(self, env, hidden=64, settings=None):
        super().__init__(env, hidden, settings)
        self.master_value = ValueNetwork(master_critic_input(env, 1.).shape[1], hidden)
        self.master_ppo = PPO(self.master, self.master_value,
                              replace(self.settings, learning_rate=1e-4))
        # The inherited checkpoint loader rejects incompatible old/new dimensions.
        self.dimensions['master_critic_remaining_time'] = True


@torch.no_grad()
def choose_epoch_master(env, learner, remaining, deterministic):
    obs, masks = master_observations(env)
    state = master_critic_input(env, remaining)
    observations, mask = torch.from_numpy(obs), torch.from_numpy(masks)
    distribution = learner.master.distribution(observations)
    latent = distribution.mean if deterministic else distribution.sample()
    log_probs, _ = learner.master.evaluate(observations, mask, latent)
    values = learner.master_value(torch.from_numpy(state)).numpy()
    actions = learner.master.actions(latent, mask).sum(0).reshape(env.member_count, 2).numpy()
    return actions, (obs, masks, latent.numpy(), log_probs.numpy(), state, values)


def run_epoch(env, learner, *, stage, steps=500, update_every=8, on_step=None):
    """Reset once, run exactly `steps` physical slots, then end the epoch.

    Both actors act in every stage. `member` updates only Member in bounded chunks;
    `master` updates only Master after the complete trajectory; `evaluation` uses
    both deterministic actors and never updates either one.
    """
    if stage not in ('member', 'master', 'evaluation'):
        raise ValueError('stage must be member, master or evaluation')
    if steps < 1 or update_every < 1:
        raise ValueError('steps and update_every must be positive')
    env.reset()
    member_chunks, member_returns, master_chunks = [], [], []
    metrics, losses = [], []
    updates = {'member_updates': 0, 'master_updates': 0}
    task_count = 0
    for index in range(steps):
        flight, master_chunk = choose_epoch_master(
            env, learner, (steps - index) / steps, deterministic=stage != 'master')
        batch = env.begin(flight)
        slot_start = batch.runtime.slot_start
        actions, chunks = plan_members(env, batch, learner,
                                        deterministic=stage != 'member', collect=stage == 'member')
        row = env.finish(batch, actions)
        metrics.append(row)
        task_count += len(actions)
        if on_step is not None:
            on_step(dict(step=index + 1, slot_start=slot_start,
                         slot_end=slot_start + env.simulator.scheduling_config.max_duration_s,
                         master_calls=1, scheduled_dags=len(batch.requests),
                         scheduled_tasks=len(actions), **row))
        if stage == 'member':
            member_chunks.extend(chunks)
            member_returns.extend([row['reward']] * sum(len(chunk[2]) for chunk in chunks))
            if (index + 1) % update_every == 0 or index + 1 == steps:
                losses.append(learner.member_ppo.update(_rollout(member_chunks, member_returns)))
                updates['member_updates'] += 1
                member_chunks.clear()
                member_returns.clear()
        elif stage == 'master':
            master_chunks.append(master_chunk)
    if stage == 'master':
        values = np.stack([chunk[-1] for chunk in master_chunks])
        rewards = np.repeat(np.array([row['reward'] for row in metrics])[:, None], env.master_count, axis=1)
        returns = gae_returns(rewards, values, [False] * (steps - 1) + [True])
        rollout = _rollout(master_chunks, returns.ravel())
        active = rollout.masks.any(axis=1)
        samples = Rollout(**{name: getattr(rollout, name)[active]
                             for name in rollout.__dataclass_fields__})
        losses.append(learner.master_ppo.update(samples))
        updates['master_updates'] = 1
    result = {name: float(np.mean([row[name] for row in metrics])) for name in metrics[0]}
    result.update(steps=steps, master_calls=steps, scheduled_tasks=task_count,
                  epoch_return=float(sum(row['reward'] for row in metrics)),
                  mean_reward=result['reward'], **updates)
    if losses:
        result.update({name: float(np.mean([loss[name] for loss in losses])) for name in losses[0]})
    return result
