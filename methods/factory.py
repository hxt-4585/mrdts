"""显式方法与组件注册；配置名称拼错时立即报错。"""

from methods.components.flight.stationary import StationaryFlight
from methods.components.ordering.adapter import ERSOrdering
from methods.components.scheduling.local import LocalScheduling
from methods.components.scheduling.owner import OwnerScheduling
from methods.components.flight.random import RandomFlight
from methods.components.scheduling.random import RandomScheduling
from methods.solutions.random.method import build_method


ORDERINGS = {"ers": ERSOrdering}
FLIGHTS = {"stationary": lambda rng: StationaryFlight(), "random": RandomFlight}
SCHEDULERS = {"local": lambda rng: LocalScheduling(), "owner": lambda rng: OwnerScheduling(),
              "random": RandomScheduling}
SOLUTIONS = {"random": build_method}
# 接入具体算法时显式注册 Trainer 类，不把评估循环伪装成训练。
TRAINERS = {}


def create_method(config, *, flight_rng, scheduling_rng):
    try:
        components = config["components"]
        return SOLUTIONS[config["solution"]](
            ORDERINGS[components["ordering"]](),
            FLIGHTS[components["flight"]](flight_rng),
            SCHEDULERS[components["scheduling"]](scheduling_rng))
    except KeyError as exc:
        raise ValueError(f"Unknown or missing method/component: {exc}") from exc


def create_trainer(config):
    name = config["solution"]
    if name not in TRAINERS:
        raise ValueError(f"No trainer registered for '{name}'. Implement and register the method's "
                         "Trainer in methods/factory.py; use experiments.run for non-learning evaluation.")
    return TRAINERS[name]()
