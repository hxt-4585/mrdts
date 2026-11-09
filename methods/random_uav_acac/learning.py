"""Macro-event GAE and recurrent PPO without duplicated history inputs.

Gamma is raised to elapsed normalized physical time; lambda is applied once per
joint decision event (paper Section 3.4). Slot terminals bootstrap at zero.
PPO minibatches pack whole slots up to a soft decision-count budget, preserving
full recurrent gradients. A single longer slot forms an oversized minibatch.
"""

import numpy as np
import torch
from torch.distributions import Categorical


def advantages(rewards, values, times, gamma, gae_lambda):
    rewards, values = np.asarray(rewards, np.float32), np.asarray(values, np.float32)
    result = np.empty_like(values)
    carry, next_value = np.zeros(values.shape[1]), np.zeros(values.shape[1])
    for index in reversed(range(len(values))):
        discount = gamma ** (times[index + 1] - times[index])
        carry = rewards[index] + discount * next_value - values[index] + discount * gae_lambda * carry
        result[index] = carry
        next_value = values[index]
    return result, result + values


def replay(model, trajectory):
    """Recompute unpadded per-agent sequences, then gather causal joint histories.

    The gather holds the latest encoding at other agents' events; it never feeds
    those duplicates back to an encoder. Attention batches over events, not time.
    """
    device = next(model.parameters()).device
    entries = [[] for _ in range(model.members)]
    owners = []
    for event_index, event in enumerate(trajectory.events):
        for row, owner in enumerate(event.owners):
            entries[owner].append((event_index, row, len(owners)))
            owners.append(owner)
    logps, entropies = [None] * len(owners), [None] * len(owners)
    histories = []
    event_indices = torch.arange(len(trajectory.events), device=device)
    for owner, records in enumerate(entries):
        if not records:
            histories.append(next(model.parameters()).new_zeros(len(trajectory.events), model.hidden))
            continue
        observations = torch.as_tensor(np.stack([trajectory.events[e].observations[r] for e, r, _ in records]),
                                       device=device)
        times = observations.new_tensor([trajectory.events[e].time for e, _, _ in records])
        actor_states = model.actor_encoders[owner].sequence(observations, times)
        critic_states = model.critic_encoders[owner].sequence(observations, times)
        available = torch.tensor([e for e, _, _ in records], device=device)
        latest = torch.searchsorted(available, event_indices, right=True)
        histories.append(torch.cat((critic_states.new_zeros(1, model.hidden), critic_states))[latest])
        masks = torch.as_tensor(np.stack([trajectory.events[e].masks[r] for e, r, _ in records]), device=device)
        actions = torch.as_tensor([int(trajectory.events[e].actions[r]) for e, r, _ in records], device=device)
        distribution = Categorical(logits=model.actors[owner](actor_states).masked_fill(~masks, -torch.inf))
        local_logp, local_entropy = distribution.log_prob(actions), distribution.entropy()
        for index, (_, _, flat_index) in enumerate(records):
            logps[flat_index], entropies[flat_index] = local_logp[index], local_entropy[index]
    values = model.central_values(torch.stack(histories, dim=1))
    return torch.stack(logps), torch.stack(entropies), values, owners


def update(model, optimizer, trajectories, *, ppo_epochs=4, minibatch_size=256,
           clip_ratio=.2, entropy_coef=.01, value_coef=.5, max_grad_norm=.5):
    trajectories = [t for t in trajectories if t.samples]
    count = sum(t.samples for t in trajectories)
    if not count:
        return dict(samples=0, batches=0, policy_loss=None, value_loss=None, entropy=None,
                    kl=None, grad_norm=None, clip_fraction=None)
    device = next(model.parameters()).device
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    actor_advantages = [np.array([t.advantages[index, owner]
                                 for index, event in enumerate(t.events) for owner in event.owners])
                        for t in trajectories]
    owners = np.array([owner for t in trajectories for e in t.events for owner in e.owners])
    all_adv = np.concatenate(actor_advantages)
    means, scales = {}, {}
    for owner in np.unique(owners):
        local = all_adv[owners == owner]
        means[owner], scales[owner] = local.mean(), local.std() + 1e-8
    totals = np.zeros(6)
    batches = 0
    for _ in range(ppo_epochs):
        groups, group, size = [], [], 0
        for index in torch.randperm(len(trajectories)).tolist():
            if group and size + trajectories[index].samples > minibatch_size:
                groups.append(group)
                group, size = [], 0
            group.append(index)
            size += trajectories[index].samples
        groups.append(group)
        for group in groups:
            policy_terms, value_terms, entropy_terms, kl_terms, clip_terms = [], [], [], [], []
            batch_count = sum(trajectories[i].samples for i in group)
            value_count = sum(len(trajectories[i].events) * int(trajectories[i].active.sum()) for i in group)
            for index in group:
                trajectory = trajectories[index]
                logp, entropy, value, local_owners = replay(model, trajectory)
                old_logp = tensor(np.concatenate([e.log_probs for e in trajectory.events]))
                advantage = tensor([(a - means[o]) / scales[o]
                                    for a, o in zip(actor_advantages[index], local_owners)])
                log_ratio = logp - old_logp
                ratio = log_ratio.exp()
                policy_terms.append(-torch.minimum(ratio * advantage,
                    ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage).sum() / batch_count)
                active = torch.as_tensor(trajectory.active, device=device)
                value_terms.append(.5 * (value[:, active] - tensor(trajectory.targets)[:, active]).square().sum()
                                   / value_count)
                entropy_terms.append(entropy.sum() / batch_count)
                kl_terms.append((ratio - 1 - log_ratio).detach().sum() / batch_count)
                clip_terms.append(((ratio - 1).abs() > clip_ratio).float().mean().detach()
                                  * trajectory.samples / batch_count)
            policy_loss, value_loss, entropy = sum(policy_terms), sum(value_terms), sum(entropy_terms)
            optimizer.zero_grad(set_to_none=True)
            (policy_loss + value_coef * value_loss - entropy_coef * entropy).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            totals += [policy_loss.item(), value_loss.item(), entropy.item(),
                       sum(kl_terms).item(), norm.item(), sum(clip_terms).item()]
            batches += 1
    return dict(samples=count, batches=batches,
                **dict(zip(('policy_loss', 'value_loss', 'entropy', 'kl', 'grad_norm', 'clip_fraction'),
                           (totals / batches).tolist())))
