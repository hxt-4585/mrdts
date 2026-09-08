"""Common CLI parsing with configuration-selected solution extensions."""

import argparse
from dataclasses import replace
import re
import tomllib

from experiments.config import DEFAULT_CONFIG, load_config, load_run_config, project_path
from methods.factory import get_solution


def training_overrides(items):
    values = {}
    for item in items:
        key, separator, value = item.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError('--training expects KEY=VALUE with a simple setting name')
        try:
            values[key] = tomllib.loads('value = '+value)['value']
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Invalid TOML value in --training {item}') from exc
    return values


def parse_config(argv=None, *, training=False):
    parser = argparse.ArgumentParser(description='MRDTS training' if training else 'MRDTS evaluation',
                                     allow_abbrev=False)
    parser.add_argument('--config')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--slots', type=int)
    parser.add_argument('--output')
    parser.add_argument('--scheduling', help='Override only the scheduling component')
    parser.add_argument('--device', help='Explicit cpu, cuda or cuda:N')
    parser.add_argument('--checkpoint', help='Saved model for evaluation')
    parser.add_argument('--episode-start', type=int, help='First workload episode ID')
    parser.add_argument('--training', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a training setting using a TOML value; repeatable')
    if training:
        parser.add_argument('--check', action='store_true', help='Validate configuration and device')
        parser.add_argument('--resume', help='Resume from a saved training checkpoint')
    # A separate parser leaves --help for the complete, solution-aware parser.
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in ('--config', '--checkpoint') + (('--resume',) if training else ()):
        selector.add_argument(flag)
    selection, _ = selector.parse_known_args(argv)
    try:
        saved = getattr(selection, 'resume', None) or selection.checkpoint
        restored = bool(saved and selection.config is None)
        config = load_run_config(saved) if restored else load_config(selection.config or DEFAULT_CONFIG)
        solution = get_solution(config.method['solution'])
        solution.add_arguments(parser, training=training)
        args = parser.parse_args(argv)
        changes = {key: getattr(args, key) for key in ('seed', 'episodes', 'slots', 'episode_start', 'device')
                   if getattr(args, key) is not None}
        for argument, field in (('output', 'output_root'), ('checkpoint', 'checkpoint'), ('resume', 'resume')):
            value = getattr(args, argument, None)
            if value:
                changes[field] = project_path(value)
        if args.training:
            changes['training'] = {**config.training, **training_overrides(args.training)}
        if args.scheduling:
            changes['method'] = {**config.method, 'components': {
                **config.method['components'], 'scheduling': args.scheduling}}
        config = solution.resolve_config(replace(config, **changes), args, training=training, restored=restored)
        return parser, args, config
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
