"""PPO owns its CLI, staged configuration and checkpoint evaluation contract."""

from dataclasses import replace
import json

from methods.solution import Solution


TRAINING_ARGUMENTS = ('member_epochs', 'master_epochs', 'eval_steps', 'test_steps',
                      'eval_every', 'update_every_steps', 'hidden', 'ppo_epochs',
                      'threads', 'log_every_steps')


class PPOSolution(Solution):
    def add_arguments(self, parser, *, training):
        if training:
            group = parser.add_argument_group('PPO training')
            for name in TRAINING_ARGUMENTS:
                group.add_argument('--'+name.replace('_', '-'), type=int)

    def resolve_config(self, config, args, *, training, restored):
        if config.method['solution'] == 'ppo_delay':
            config = replace(config, method={**config.method, 'solution': 'ppo'},
                             name='ppo' if config.name == 'ppo_delay' else config.name)
        if not training:
            if restored:
                return replace(config,
                               episodes=1 if args.episodes is None else config.episodes,
                               slots=config.training.get('test_steps', config.slots) if args.slots is None else config.slots,
                               episode_start=1 if args.episode_start is None else config.episode_start)
            return config
        overrides = {name: getattr(args, name) for name in TRAINING_ARGUMENTS
                     if getattr(args, name) is not None}
        settings = {**config.training, **overrides}
        member, master = settings.get('member_epochs', 100), settings.get('master_epochs', 30)
        explicit = set(overrides) | {item.split('=', 1)[0] for item in args.training}
        if args.episodes is not None and not explicit.intersection(('member_epochs', 'master_epochs')):
            member = min(member, args.episodes)
            master = args.episodes - member
            settings.update(member_epochs=member, master_epochs=master)
        return replace(config, training=settings,
                       episodes=member+master if args.episodes is None else config.episodes)

    def validate_components(self, components):
        if components != dict(ordering='ers', flight='ppo_master', scheduling='ppo_member'):
            raise ValueError('ppo requires ers / ppo_master / ppo_member')

    def validate_config(self, config, *, training):
        if training:
            if config.checkpoint is not None:
                raise ValueError('Use --resume for PPO training, or experiments.run --checkpoint for evaluation')
            from .settings import TrainingSettings
            TrainingSettings.from_config(config)
            return
        if config.checkpoint is None:
            raise ValueError('PPO evaluation requires --checkpoint')
        from .checkpoint import read_checkpoint
        from .settings import environment_signature
        from methods.learning.device import check_device
        state = read_checkpoint(config.checkpoint)
        requested_ids = set(range(config.episode_start, config.episode_start + config.episodes))
        test_ids = set(state['metadata']['signature']['splits']['test'])
        if not requested_ids <= test_ids:
            raise ValueError('PPO evaluation must use the held-out test split: --episodes 1 --episode-start 1')
        if state['metadata']['signature']['environment'] != json.loads(json.dumps(environment_signature(config))):
            raise ValueError('Evaluation seed or physical configuration differs from the checkpoint')
        check_device(config.device)

    def build_method(self, config, ordering, flight, scheduling, *, checkpoint, device):
        from .method import PPOMethod
        return PPOMethod(ordering=ordering, flight=flight, scheduling=scheduling,
                         checkpoint=checkpoint, device=device)

    def create_trainer(self):
        from .trainer import PPOTrainer
        return PPOTrainer()
