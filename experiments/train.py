"""训练入口：命令行和 PyCharm 调用同一个方法 Trainer。"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.cli import parse_config
from experiments.randomness import RandomStreams
from methods.factory import create_method, create_trainer
from methods.learning.device import check_device


def main(argv=None):
    parser, args, config = parse_config(argv, training=True)
    try:
        if args.check:
            if config.method['solution'] == 'ppo_delay':
                from methods.solutions.ppo_delay.settings import TrainingSettings
                TrainingSettings.from_config(config)
            streams = RandomStreams.from_config(config)
            create_method(config.method, flight_rng=streams.flight, scheduling_rng=streams.scheduling)
            device, details = check_device(config.device)
            print(details)
            print("Configuration/device check passed. This command does not train a model.")
            return device
        trainer = create_trainer(config.method)
        device, details = check_device(config.device)
        print(details)
        return trainer.train(config, device)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
