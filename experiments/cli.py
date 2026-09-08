"""命令行与 PyCharm 入口共用参数处理。"""

import argparse
from dataclasses import replace

from experiments.config import DEFAULT_CONFIG, load_config, load_run_config, project_path


def parse_config(argv=None, *, training=False):
    parser = argparse.ArgumentParser(description="MRDTS training" if training else "MRDTS evaluation")
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--output")
    parser.add_argument("--scheduling", help="Override only the scheduling component")
    parser.add_argument('--device', help='Explicit cpu, cuda or cuda:N')
    parser.add_argument('--checkpoint', help='PPO checkpoint for deterministic evaluation')
    parser.add_argument('--episode-start', type=int, help='First workload episode ID; PPO test uses 1')
    if training:
        parser.add_argument("--check", action="store_true", help="Validate config and execute a device check")
        parser.add_argument('--resume', help='Resume migrated checkpoints/latest.pt')
        for name in ('member-epochs', 'master-epochs', 'eval-steps', 'test-steps', 'eval-every',
                     'update-every-steps', 'hidden', 'ppo-epochs', 'threads', 'log-every-steps'):
            parser.add_argument('--'+name, type=int)
    args = parser.parse_args(argv)
    try:
        saved_checkpoint = getattr(args, 'resume', None) or args.checkpoint
        if saved_checkpoint and args.config is None:
            config = load_run_config(saved_checkpoint)
            if not training:
                config = replace(config, episodes=1, slots=config.training.get('test_steps', config.slots),
                                 episode_start=1)
        else:
            config = load_config(args.config or DEFAULT_CONFIG)
        changes = {key: getattr(args, key) for key in ("seed", "episodes", "slots")
                   if getattr(args, key) is not None}
        if args.output:
            changes["output_root"] = project_path(args.output)
        if args.device:
            changes["device"] = args.device
        if args.checkpoint:
            changes['checkpoint'] = project_path(args.checkpoint)
        if args.episode_start is not None:
            changes['episode_start'] = args.episode_start
        if training:
            if args.resume:
                changes['resume'] = project_path(args.resume)
            overrides = {name:getattr(args,name) for name in ('member_epochs','master_epochs','eval_steps','test_steps',
                         'eval_every','update_every_steps','hidden','ppo_epochs','threads','log_every_steps')
                         if getattr(args,name) is not None}
            if overrides:
                changes['training'] = {**config.training, **overrides}
            if config.method['solution'] == 'ppo_delay':
                settings = changes.get('training', config.training)
                member, master = settings.get('member_epochs',100), settings.get('master_epochs',30)
                if args.episodes is not None and not any(name in overrides for name in ('member_epochs','master_epochs')):
                    member = min(member,args.episodes)
                    master = args.episodes-member
                    changes['training'] = {**settings,'member_epochs':member,'master_epochs':master}
                if args.episodes is None:
                    changes['episodes'] = member+master
        if args.scheduling:
            changes["method"] = {**config.method, "components": {
                **config.method["components"], "scheduling": args.scheduling}}
        return parser, args, replace(config, **changes)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
