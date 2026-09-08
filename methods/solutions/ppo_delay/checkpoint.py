"""Versioned Kbit checkpoints; full epoch state is atomically replaced."""

from dataclasses import asdict
from pathlib import Path

import torch


FORMAT_VERSION = 3
SOURCE_COMMIT = 'c47059dee760e0095b7d135a856a114fa63e1a5e'
NETWORKS = ('member', 'master', 'member_value', 'master_value')


def read_checkpoint(path):
    state = torch.load(Path(path), map_location='cpu', weights_only=True)
    if state.get('format_version') != FORMAT_VERSION or state.get('feature_schema') != 'kbit_23_v1':
        raise ValueError('Checkpoint must use the migrated Kbit PPO schema; legacy KB checkpoints are incompatible')
    return state


def save_learner(learner, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(format_version=FORMAT_VERSION, feature_schema='kbit_23_v1', source_commit=SOURCE_COMMIT,
                 hidden=learner.hidden, settings=asdict(learner.settings), dimensions=learner.dimensions,
                 metadata=metadata, rng=torch.get_rng_state(),
                 selected_checkpoints=getattr(learner, 'selected_checkpoints', {}),
                 cuda_rng=torch.cuda.get_rng_state_all() if learner.device.type == 'cuda' else [])
    for name in NETWORKS:
        state[name] = getattr(learner,name).state_dict()
    for name in ('member_ppo','master_ppo'):
        ppo = getattr(learner,name)
        state[name] = dict(actor=ppo.actor_optimizer.state_dict(), critic=ppo.critic_optimizer.state_dict())
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_learner(learner, path, optimizers=True, restore_rng=True):
    state = read_checkpoint(path)
    if state['dimensions'] != learner.dimensions:
        raise ValueError('Checkpoint environment/network dimensions do not match')
    for name in NETWORKS:
        getattr(learner,name).load_state_dict(state[name])
    if optimizers:
        if state['settings'] != asdict(learner.settings):
            raise ValueError('Checkpoint PPO settings differ from this learner')
        for name in ('member_ppo','master_ppo'):
            ppo = getattr(learner,name)
            ppo.actor_optimizer.load_state_dict(state[name]['actor'])
            ppo.critic_optimizer.load_state_dict(state[name]['critic'])
        learner.selected_checkpoints = state.get('selected_checkpoints', {})
    if restore_rng:
        torch.set_rng_state(state['rng'])
        if learner.device.type == 'cuda' and state['cuda_rng']:
            if len(state['cuda_rng']) != torch.cuda.device_count():
                raise ValueError('Saved CUDA RNG topology differs; exact resume needs the same device topology')
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['metadata']
