"""Reusable MLP and scalar value network; no UAV dependencies."""
from torch import nn

def mlp(inputs, outputs, hidden=64, output_gain=1.):
    network = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=2 ** .5)
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(network[-1].weight, gain=output_gain)
    return network


class ValueNetwork(nn.Module):
    def __init__(self, inputs, hidden=64):
        super().__init__()
        self.network = mlp(inputs, 1, hidden)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -.8)

    def forward(self, observations):
        return self.network(observations).squeeze(-1)
