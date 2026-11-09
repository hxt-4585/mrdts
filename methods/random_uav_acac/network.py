"""Jung et al. (ICML 2025), Sections 3.2–3.3, adapted to Member-local rewards.

Actor and critic histories have separate per-agent encoders. No encoder is fed
duplicated observations for an executing agent. The centralized value head has
one output per Member because this environment does not use a shared reward.
https://proceedings.mlr.press/v267/jung25a.html
"""

import math

import torch
from torch import nn


class HistoryEncoder(nn.Module):
    def __init__(self, obs_dim, hidden, time_dim):
        super().__init__()
        self.register_buffer('frequencies', torch.exp(
            -math.log(10000.) * torch.arange(0, time_dim, 2) / time_dim))
        self.input = nn.Sequential(nn.Linear(obs_dim + time_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, observation, time, previous):
        phase = self.frequencies * time
        encoding = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten()
        inputs = self.input(torch.cat((observation, encoding))).view(1, 1, -1)
        _, hidden = self.gru(inputs, previous.view(1, 1, -1))
        return hidden.view(-1)

    def sequence(self, observations, times):
        """Encode only this agent's actual observations, without temporal padding."""
        phase = times[:, None] * self.frequencies
        encoding = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)
        inputs = self.input(torch.cat((observations, encoding), dim=-1))
        output, _ = self.gru(inputs.unsqueeze(0))
        return output.squeeze(0)


class ACAC(nn.Module):
    def __init__(self, obs_dim, members, hidden=128, time_dim=16, attention_heads=4):
        super().__init__()
        self.members, self.hidden = members, hidden
        self.actor_encoders = nn.ModuleList(HistoryEncoder(obs_dim, hidden, time_dim) for _ in range(members))
        self.critic_encoders = nn.ModuleList(HistoryEncoder(obs_dim, hidden, time_dim) for _ in range(members))
        self.actors = nn.ModuleList(nn.Linear(hidden, members + 2) for _ in range(members))
        self.attention = nn.MultiheadAttention(hidden, attention_heads, dropout=0., batch_first=True)
        self.value_body = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.value_head = nn.Linear(hidden, members)
        for actor in self.actors:
            nn.init.orthogonal_(actor.weight, .01)
            nn.init.zeros_(actor.bias)

    def initial_state(self):
        parameter = next(self.parameters())
        return ([parameter.new_zeros(self.hidden) for _ in range(self.members)],
                [parameter.new_zeros(self.hidden) for _ in range(self.members)])

    def advance(self, observations, time, state):
        actor_states, critic_states = (list(part) for part in state)
        logits = {}
        for owner, observation in observations.items():
            actor_states[owner] = self.actor_encoders[owner](observation, time, actor_states[owner])
            critic_states[owner] = self.critic_encoders[owner](observation, time, critic_states[owner])
            logits[owner] = self.actors[owner](actor_states[owner])
        histories = torch.stack(critic_states).unsqueeze(0)
        values = self.central_values(histories).squeeze(0)
        return logits, values, (actor_states, critic_states)

    def central_values(self, histories):
        context, _ = self.attention(histories, histories, histories, need_weights=False)
        return self.value_head(self.value_body(context.mean(dim=1)))
