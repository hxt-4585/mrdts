"""Adapt older Member policy weights to the observation used for inference."""

import torch


def without_legacy_member_fields(weight, count, *, energy=False, dag_share=False, time_dim=0):
    prefix = 4 + (5 if energy else 4) * count
    user_start = prefix + count + 2
    legacy_width = user_start + 2 + dag_share + time_dim
    if weight.shape[1] != legacy_width:
        return weight
    return torch.cat((weight[:, :prefix], weight[:, user_start:user_start + 2],
                      weight[:, user_start + 2 + dag_share:]), dim=1)


def load_member_weights(model, weights, count, **layout):
    """Load policy weights, dropping removed feature columns from legacy inputs."""
    adapted = weights.copy()
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and name in weights and weights[name].shape != parameter.shape:
            adapted[name] = without_legacy_member_fields(weights[name], count, **layout)
    model.load_state_dict(adapted)
