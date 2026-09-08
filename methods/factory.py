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
def ppo_flight(rng):
    from methods.components.flight.ppo import PPOFlight
    return PPOFlight()


def ppo_scheduling(rng):
    from methods.components.scheduling.ppo import PPOScheduling
    return PPOScheduling()


def ppo_method(ordering, flight, scheduling):
    from methods.solutions.ppo_delay.method import build_method
    return build_method(ordering, flight, scheduling)


def ppo_trainer():
    from methods.solutions.ppo_delay.trainer import PPODelayTrainer
    return PPODelayTrainer()


FLIGHTS['ppo_master'] = ppo_flight
SCHEDULERS['ppo_member'] = ppo_scheduling
SOLUTIONS = {"random": build_method, "ppo_delay": ppo_method}
TRAINERS = {'ppo_delay': ppo_trainer}

# Result names identify algorithms; component roles are already given by their position.
COMPONENT_LABELS = {'ppo_master': 'ppo', 'ppo_member': 'ppo'}


def validate_method(config):
    try:
        solution, components = config['solution'], config['components']
        if solution not in SOLUTIONS:
            raise ValueError(f'Unknown solution: {solution}')
        for key, registry in (('ordering', ORDERINGS), ('flight', FLIGHTS), ('scheduling', SCHEDULERS)):
            if components[key] not in registry:
                raise ValueError(f'Unknown {key}: {components[key]}')
        if solution == 'ppo_delay' and components != dict(ordering='ers', flight='ppo_master', scheduling='ppo_member'):
            raise ValueError('ppo_delay requires ers / ppo_master / ppo_member')
        if solution == 'random' and (components['flight'] == 'ppo_master' or components['scheduling'] == 'ppo_member'):
            raise ValueError('Learned PPO components require the ppo_delay observation context')
    except KeyError as exc:
        raise ValueError(f'Missing method/component: {exc}') from exc


def create_method(config, *, flight_rng, scheduling_rng, checkpoint=None, device='cpu'):
    validate_method(config)
    try:
        components = config["components"]
        method = SOLUTIONS[config["solution"]](
            ORDERINGS[components["ordering"]](),
            FLIGHTS[components["flight"]](flight_rng),
            SCHEDULERS[components["scheduling"]](scheduling_rng))
        if config['solution'] == 'ppo_delay':
            method.checkpoint, method.device = checkpoint, device
        return method
    except KeyError as exc:
        raise ValueError(f"Unknown or missing method/component: {exc}") from exc


def create_trainer(config):
    name = config["solution"]
    if name not in TRAINERS:
        raise ValueError(f"No trainer registered for '{name}'. Implement and register the method's "
                         "Trainer in methods/factory.py; use experiments.run for non-learning evaluation.")
    return TRAINERS[name]()
