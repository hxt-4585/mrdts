"""Non-learning solution integration."""

from methods.solution import Solution


class RandomSolution(Solution):
    def build_method(self, config, ordering, flight, scheduling, *, checkpoint, device):
        if checkpoint is not None:
            raise ValueError('This solution does not accept a checkpoint')
        from .method import build_method
        return build_method(ordering, flight, scheduling)

    def validate_components(self, components):
        from methods.factory import LEARNED_COMPONENTS
        if set(components.values()) & LEARNED_COMPONENTS:
            raise ValueError('Learned components require their solution observation context')
