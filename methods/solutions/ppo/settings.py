"""Validated training settings and immutable epoch/split contract."""

from dataclasses import asdict, dataclass, fields
import json

from experiments.scene import resolved_settings, settings_snapshot
from methods.learning.algorithms.ppo import PPOSettings


@dataclass(frozen=True)
class TrainingSettings:
    member_epochs: int = 100
    master_epochs: int = 30
    update_every_steps: int = 8
    eval_steps: int = 500
    test_steps: int = 500
    eval_every: int = 10
    hidden: int = 64
    ppo_epochs: int = 4
    threads: int = 1
    log_every_steps: int = 25

    @classmethod
    def from_config(cls, config):
        unknown = set(config.training) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown PPO training settings: {sorted(unknown)}')
        result = cls(**config.training)
        for field in fields(cls):
            value = getattr(result, field.name)
            minimum = 0 if field.name in ('member_epochs', 'master_epochs') else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{field.name} must be an integer >= {minimum}')
        if result.member_epochs + result.master_epochs != config.episodes:
            raise ValueError('experiment.episodes must equal member_epochs + master_epochs')
        return result

    def ppo(self):
        return PPOSettings(epochs=self.ppo_epochs)


def environment_signature(config):
    return dict(seed=config.seed, scenes=settings_snapshot(resolved_settings(config)))


def training_signature(config, settings):
    value = dict(environment=environment_signature(config), slots=config.slots,
                 method=config.method, settings={key:value for key,value in asdict(settings).items()
                                                 if key not in ('member_epochs','master_epochs','log_every_steps')},
                 splits=dict(validation=[0], test=[1], training_start=2), feature_schema='kbit_23_v1')
    return json.loads(json.dumps(value, default=str))
