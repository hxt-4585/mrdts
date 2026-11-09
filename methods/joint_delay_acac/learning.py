"""Independent recurrent actors and an asynchronous attention critic."""

import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class History(nn.Module):
    def __init__(self, obs_dim, hidden, time_dim):
        super().__init__()
        dimensions = torch.arange(time_dim)
        self.register_buffer('frequencies', torch.exp(-math.log(10000.) * (2 * (dimensions // 2)) / time_dim))
        self.register_buffer('cosine_dimensions', dimensions % 2 == 1)
        self.input = nn.Sequential(nn.Linear(obs_dim + time_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def encode_time(self, times):
        angles = times[:, None] * self.frequencies
        return torch.where(self.cosine_dimensions, angles.cos(), angles.sin())

    def forward(self, observation, time, hidden):
        encoded = self.encode_time(observation.new_tensor([time]))
        sequence = self.input(torch.cat((observation.reshape(1, -1), encoded), -1)).unsqueeze(0)
        output, _ = self.gru(sequence, hidden.reshape(1, 1, -1))
        return output[0, 0]

    def sequence(self, observations, times):
        features = self.input(torch.cat((observations, self.encode_time(times)), -1))
        output, _ = self.gru(features[None])
        return output[0]


class ACAC(nn.Module):
    def __init__(self, members, obs_dim, config):
        super().__init__()
        self.members, self.hidden = members, config['hidden']
        self.actor_encoders = nn.ModuleList(History(obs_dim, self.hidden, config['time_dim']) for _ in range(members))
        self.critic_encoders = nn.ModuleList(History(obs_dim, self.hidden, config['time_dim']) for _ in range(members))
        self.actors = nn.ModuleList(nn.Linear(self.hidden, members + 2) for _ in range(members))
        for actor in self.actors:
            nn.init.orthogonal_(actor.weight, .01)
            nn.init.zeros_(actor.bias)
        self.attention = nn.MultiheadAttention(self.hidden, config['attention_heads'], batch_first=True)
        self.value_heads = nn.ModuleList(nn.Linear(2 * self.hidden, 1) for _ in range(members))

    def initial_state(self):
        zero = next(self.parameters()).new_zeros(self.members, self.hidden)
        return zero, zero.clone()

    def central_values(self, histories):
        context, _ = self.attention(histories, histories, histories, need_weights=False)
        features = torch.cat((histories, context), dim=-1)
        return torch.cat([head(features[:, owner]) for owner, head in enumerate(self.value_heads)], -1)

    def advance(self, observations, time, state):
        actor_state, critic_state = list(state[0].unbind()), list(state[1].unbind())
        logits = {}
        for owner, observation in observations.items():
            actor_state[owner] = self.actor_encoders[owner](observation, time, actor_state[owner])
            critic_state[owner] = self.critic_encoders[owner](observation, time, critic_state[owner])
            logits[owner] = self.actors[owner](actor_state[owner])
        updated = torch.stack(actor_state), torch.stack(critic_state)
        return logits, self.central_values(updated[1][None])[0], updated


def make_model(member_count, obs_dim, config, device):
    return ACAC(member_count, obs_dim, config).to(device)


def make_optimizers(model, config):
    return torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-5)


def advantages(rewards, values, times, gamma, gae_lambda):
    estimates = np.asarray(values, dtype=np.float32)
    advantage = np.zeros_like(estimates)
    future = np.zeros(estimates.shape[1], dtype=np.float32)
    next_value = np.zeros_like(future)
    for event in range(len(estimates) - 1, -1, -1):
        discount = gamma ** (times[event + 1] - times[event])
        residual = rewards[event] + discount * next_value - estimates[event]
        future = residual + discount * gae_lambda * future
        advantage[event] = future
        next_value = estimates[event]
    return advantage, estimates + advantage


def replay(model, trajectory):
    """Encode each owner's unpadded sequence once, gathering only causal states."""
    parameter = next(model.parameters())
    device, event_count = parameter.device, len(trajectory.events)
    records = [[] for _ in range(model.members)]
    flat_owners = []
    for index, event in enumerate(trajectory.events):
        for row, owner in enumerate(event.owners):
            records[owner].append((index, row, len(flat_owners)))
            flat_owners.append(owner)
    logps, entropies = [None] * len(flat_owners), [None] * len(flat_owners)
    histories = []
    event_range = torch.arange(event_count, device=device)
    for owner, sequence in enumerate(records):
        if not sequence:
            histories.append(parameter.new_zeros(event_count, model.hidden))
            continue
        observations = torch.as_tensor(np.stack([trajectory.events[i].observations[r] for i, r, _ in sequence]), device=device)
        times = observations.new_tensor([trajectory.events[i].time for i, _, _ in sequence])
        local_actor = model.actor_encoders[owner].sequence(observations, times)
        local_critic = model.critic_encoders[owner].sequence(observations, times)
        indices = torch.tensor([i for i, _, _ in sequence], device=device)
        causal_index = torch.searchsorted(indices, event_range, right=True)
        histories.append(torch.cat((parameter.new_zeros(1, model.hidden), local_critic))[causal_index])
        masks = torch.as_tensor(np.stack([trajectory.events[i].masks[r] for i, r, _ in sequence]), device=device)
        actions = torch.tensor([int(trajectory.events[i].actions[r]) for i, r, _ in sequence], device=device)
        distribution = Categorical(logits=model.actors[owner](local_actor).masked_fill(~masks, -torch.inf))
        owner_logp, owner_entropy = distribution.log_prob(actions), distribution.entropy()
        for j, (_, _, flat) in enumerate(sequence):
            logps[flat], entropies[flat] = owner_logp[j], owner_entropy[j]
    if not flat_owners:
        return parameter.new_empty(0), parameter.new_empty(0), parameter.new_empty(0, model.members), flat_owners
    values = model.central_values(torch.stack(histories, 1))
    return torch.stack(logps), torch.stack(entropies), values, flat_owners


def update(model, optimizers, trajectories, config):
    slots = [slot for slot in trajectories if slot.samples]
    samples = sum(slot.samples for slot in slots)
    if not samples:
        return dict(samples=0, policy_loss=0., value_loss=0., entropy=0., kl=0.)
    device = next(model.parameters()).device
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    old_logps = [tensor(np.concatenate([e.log_probs for e in slot.events])) for slot in slots]
    actor_advantages = [np.array([slot.advantages[i, owner] for i, e in enumerate(slot.events) for owner in e.owners]) for slot in slots]
    actor_owners = [np.array([owner for e in slot.events for owner in e.owners]) for slot in slots]
    all_adv, all_owners = np.concatenate(actor_advantages), np.concatenate(actor_owners)
    normalized = [adv.copy() for adv in actor_advantages]
    for owner in np.unique(all_owners):
        local = all_adv[all_owners == owner]
        for adv, owners in zip(normalized, actor_owners):
            adv[owners == owner] = (adv[owners == owner] - local.mean()) / (local.std() + 1e-8)
    normalized = [tensor(adv) for adv in normalized]
    targets = [tensor(slot.targets) for slot in slots]
    active = [torch.as_tensor(slot.active, device=device) for slot in slots]
    statistics = []
    for _ in range(config['ppo_epochs']):
        batches, batch, size = [], [], 0
        for index in torch.randperm(len(slots)).tolist():
            if batch and size + slots[index].samples > config['minibatch_size']:
                batches.append(batch)
                batch, size = [], 0
            batch.append(index)
            size += slots[index].samples
        batches.append(batch)
        for batch in batches:
            actor_count = sum(slots[i].samples for i in batch)
            critic_count = sum(len(slots[i].events) * int(slots[i].active.sum()) for i in batch)
            policy_terms, value_terms, entropy_terms, kl_terms = [], [], [], []
            for i in batch:
                logp, entropy, value, _ = replay(model, slots[i])
                log_ratio = logp - old_logps[i]
                ratio = log_ratio.exp()
                clipped = ratio.clamp(1 - config['clip_ratio'], 1 + config['clip_ratio'])
                policy_terms.append(-torch.minimum(ratio * normalized[i], clipped * normalized[i]).sum() / actor_count)
                value_terms.append(.5 * (value[:, active[i]] - targets[i][:, active[i]]).square().sum() / critic_count)
                entropy_terms.append(entropy.sum() / actor_count)
                kl_terms.append((ratio - 1 - log_ratio).sum() / actor_count)
            policy, value, entropy, kl = map(sum, (policy_terms, value_terms, entropy_terms, kl_terms))
            optimizers.zero_grad(set_to_none=True)
            (policy + config['value_coef'] * value - config['entropy_coef'] * entropy).backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizers.step()
            statistics.append(torch.stack((policy, value, entropy, kl)).detach())
    result = torch.stack(statistics).mean(0).cpu().tolist()
    return dict(samples=samples, **dict(zip(('policy_loss', 'value_loss', 'entropy', 'kl'), result)))
