"""命令行与 PyCharm 入口共用参数处理。"""

import argparse
from dataclasses import replace

from experiments.config import DEFAULT_CONFIG, load_config, project_path


def parse_config(argv=None, *, training=False):
    parser = argparse.ArgumentParser(description="MRDTS training" if training else "MRDTS evaluation")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--output")
    parser.add_argument("--scheduling", help="Override only the scheduling component")
    if training:
        parser.add_argument("--device", help="cuda (default), cuda:0, or explicitly cpu")
        parser.add_argument("--check", action="store_true", help="Validate config and execute a device check")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        changes = {key: getattr(args, key) for key in ("seed", "episodes", "slots")
                   if getattr(args, key) is not None}
        if args.output:
            changes["output_root"] = project_path(args.output)
        if training and args.device:
            changes["device"] = args.device
        if args.scheduling:
            changes["method"] = {**config.method, "components": {
                **config.method["components"], "scheduling": args.scheduling}}
        return parser, args, replace(config, **changes)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
