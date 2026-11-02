"""仅聚合标量的 PPO 诊断；不采样随机数、不保留计算图或历史 rollout。"""

import math

import torch


FIELDS = ('active_samples', 'reward_mean', 'reward_std', 'advantage_mean', 'advantage_std',
          'return_mean', 'return_std', 'value_mean', 'value_std', 'explained_variance_before',
          'grad_norm', 'grad_clip_fraction', 'grad_scale', 'actor_grad_norm', 'critic_grad_norm',
          'final_kl', 'final_clip_fraction', 'ratio_p05', 'ratio_p95', 'explained_variance_after',
          'value_mse_after', 'action_mean', 'action_std', 'action_saturation_fraction',
          'active_dims_mean', 'log_std_mean', 'log_std_change', 'entropy_per_dim',
          'categorical_entropy', 'categorical_entropy_normalized')


def empty_stats(policy):
    stats = dict.fromkeys(FIELDS)
    stats['active_samples'] = 0
    if not policy.continuous:
        stats.update({f'action_{i}_fraction': None for i in range(policy.actor[-1].out_features)})
    return stats


def explained_variance(values, targets):
    variance = targets.var(unbiased=False)
    # 恒定回报的解释方差无定义；CSV 留空，避免把无定义误作拟合良好。
    return (1 - (targets - values).var(unbiased=False) / variance).item() if variance.item() > 0 else None


@torch.no_grad()
def rollout_stats(policy, samples, actions, masks, targets, advantages, active):
    stats = empty_stats(policy)
    stats['active_samples'] = int(active.sum())
    rewards = targets.new_tensor([s.reward for s in samples])
    values = targets.new_tensor([s.value for s in samples])
    for name, data in (('reward', rewards), ('advantage', advantages[active]),
                       ('return', targets), ('value', values)):
        stats[name + '_mean'] = data.mean().item()
        stats[name + '_std'] = data.std(unbiased=False).item()
    stats['explained_variance_before'] = explained_variance(values, targets)
    stats['active_dims_mean'] = masks[active].sum(-1).float().mean().item()
    if policy.continuous:
        bounded = actions.tanh()[masks]
        stats.update(action_mean=bounded.mean().item(), action_std=bounded.std(unbiased=False).item(),
                     action_saturation_fraction=(bounded.abs() > .95).float().mean().item())
    else:
        fractions = torch.bincount(actions, minlength=masks.shape[-1]).float() / len(samples)
        stats.update({f'action_{i}_fraction': value for i, value in enumerate(fractions.tolist())})
    return stats


@torch.no_grad()
def final_stats(policy, obs, state, actions, masks, old_logp, targets, active, clip_ratio):
    # 不调用 score：连续策略的 score 会为熵采样，改变后续训练随机流。
    distribution = policy.distribution(obs, masks)
    if policy.continuous:
        jacobian = 2 * (math.log(2.) - actions
                        - torch.nn.functional.softplus(-2 * actions))
        logp = ((distribution.log_prob(actions) - jacobian) * masks).sum(-1)
        extra = dict(log_std_mean=policy.log_std.clamp(-5., 2.)[masks.any(0)].mean().item())
    else:
        logp = distribution.log_prob(actions)
        entropy = distribution.entropy()
        counts = masks.sum(-1)
        # 只有一个合法动作时归一化熵定义为 0。
        normalized = entropy / counts.clamp_min(2).float().log()
        extra = dict(categorical_entropy=entropy.mean().item(),
                     categorical_entropy_normalized=normalized.mean().item())
    logratio = (logp - old_logp)[active]
    ratio = logratio.exp()
    quantiles = torch.quantile(ratio, ratio.new_tensor([.05, .95])).tolist()
    values = policy.critic(state).squeeze(-1)
    return dict(final_kl=(ratio - 1 - logratio).mean().item(),
                final_clip_fraction=((ratio - 1).abs() > clip_ratio).float().mean().item(),
                ratio_p05=quantiles[0], ratio_p95=quantiles[1],
                explained_variance_after=explained_variance(values, targets),
                value_mse_after=(values - targets).square().mean().item(), **extra)


def grad_norm(parameters):
    return torch.stack([p.grad.detach().norm() for p in parameters]).norm()
