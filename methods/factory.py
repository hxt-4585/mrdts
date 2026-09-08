"""Explicit registration; experiment entry points know no solution names."""

from importlib import import_module

from methods.components.flight.stationary import StationaryFlight
from methods.components.ordering.adapter import ERSOrdering
from methods.components.scheduling.local import LocalScheduling
from methods.components.scheduling.owner import OwnerScheduling
from methods.components.flight.random import RandomFlight
from methods.components.scheduling.random import RandomScheduling


def ppo_flight(rng):
    from methods.components.flight.ppo import PPOFlight
    return PPOFlight()


def ppo_scheduling(rng):
    from methods.components.scheduling.ppo import PPOScheduling
    return PPOScheduling()


ORDERINGS = {'ers': ERSOrdering}
FLIGHTS = {'stationary': lambda rng: StationaryFlight(), 'random': RandomFlight, 'ppo_master': ppo_flight}
SCHEDULERS = {'local': lambda rng: LocalScheduling(), 'owner': lambda rng: OwnerScheduling(),
              'random': RandomScheduling, 'ppo_member': ppo_scheduling}
LEARNED_COMPONENTS = {'ppo_master', 'ppo_member'}
COMPONENT_LABELS = {'ppo_master': 'ppo', 'ppo_member': 'ppo'}
# Lazy import keeps non-learning entry points independent of the ML runtime.
SOLUTIONS = {'random': 'methods.solutions.random.solution:RandomSolution',
             'ppo': 'methods.solutions.ppo.solution:PPOSolution'}
# Read historical configurations without retaining a second implementation.
SOLUTION_ALIASES = {'ppo_delay': 'ppo'}


def get_solution(name):
    try:
        constructor = SOLUTIONS[SOLUTION_ALIASES.get(name, name)]
    except KeyError as exc:
        raise ValueError(f'Unknown solution: {name}') from exc
    if isinstance(constructor, str):
        module, attribute = constructor.split(':')
        constructor = getattr(import_module(module), attribute)
    return constructor()


def validate_method(config):
    try:
        solution = get_solution(config['solution'])
        components = config['components']
        for key, registry in (('ordering', ORDERINGS), ('flight', FLIGHTS), ('scheduling', SCHEDULERS)):
            if components[key] not in registry:
                raise ValueError(f'Unknown {key}: {components[key]}')
        solution.validate_components(components)
    except KeyError as exc:
        raise ValueError(f'Missing method/component: {exc}') from exc
    return solution


def create_method(config, *, flight_rng, scheduling_rng, checkpoint=None, device='cpu'):
    solution = validate_method(config)
    components = config['components']
    return solution.build_method(
        config, ORDERINGS[components['ordering']](), FLIGHTS[components['flight']](flight_rng),
        SCHEDULERS[components['scheduling']](scheduling_rng), checkpoint=checkpoint, device=device)


def create_trainer(config):
    return validate_method(config).create_trainer()
