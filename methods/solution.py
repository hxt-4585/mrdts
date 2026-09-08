"""Extension points between experiment orchestration and a complete method."""

from methods.compose import CompositeMethod


class Solution:
    """Override only the capabilities a solution needs; no training by default."""

    def add_arguments(self, parser, *, training):
        """Declare solution-specific CLI flags after configuration selection."""

    def resolve_config(self, config, args, *, training, restored):
        """Resolve solution defaults after common CLI overrides."""
        return config

    def validate_components(self, components):
        """Check solution-specific combinations before creating a run."""

    def validate_config(self, config, *, training):
        """Check model/config compatibility before creating artifacts."""
        if config.checkpoint is not None or config.resume is not None:
            raise ValueError('This solution does not accept a checkpoint or resume')

    def build_method(self, config, ordering, flight, scheduling, *, checkpoint, device):
        """Return an object providing run_slot(scene, workload) -> SlotOutcome."""
        if checkpoint is not None:
            raise ValueError('This solution does not accept a checkpoint')
        return CompositeMethod(ordering, flight, scheduling)

    def create_trainer(self):
        raise ValueError('No trainer registered for this solution; use experiments.run for non-learning evaluation')
