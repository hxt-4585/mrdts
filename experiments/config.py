"""实验配置与项目路径；所有默认路径独立于启动目录。"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "experiments" / "random.toml"


def project_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    episodes: int
    slots: int
    users: int
    dag_nodes: int
    output_root: Path
    device: str
    method: dict
    training: dict = field(default_factory=dict)
    checkpoint: Path | None = None
    resume: Path | None = None
    episode_start: int = 0

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError("Experiment name must contain only letters, numbers, '_' or '-'")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("episodes", "slots", "users", "dag_nodes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.episode_start, bool) or not isinstance(self.episode_start, int) or self.episode_start < 0:
            raise ValueError('episode_start must be a nonnegative integer')


def load_config(path=DEFAULT_CONFIG):
    with project_path(path).open("rb") as stream:
        data = tomllib.load(stream)
    experiment = data["experiment"]
    with project_path(experiment["method_config"]).open("rb") as stream:
        method = tomllib.load(stream)["method"]
    overrides = data.get("components", {})
    method["components"] = {**method["components"], **overrides}
    return ExperimentConfig(
        name=experiment["name"], seed=experiment["seed"], episodes=experiment["episodes"],
        slots=experiment["slots"], users=experiment["users"], dag_nodes=experiment["dag_nodes"],
        output_root=project_path(experiment.get("output_root", "results")),
        device=experiment.get("device", "cuda"), method=method, training=data.get("training", {}),
        checkpoint=project_path(experiment['checkpoint']) if experiment.get('checkpoint') else None,
        episode_start=experiment.get('episode_start', 0))


def load_run_config(checkpoint):
    """Restore the resolved experiment saved alongside a migrated checkpoint."""
    path = project_path(checkpoint).parent.parent / 'config.json'
    with path.open(encoding='utf-8') as stream:
        saved = json.load(stream)
    values = saved['experiment'].copy()
    values.update(output_root=project_path(values['output_root']), checkpoint=None, resume=None)
    values['training'] = saved.get('resolved_training', values.get('training', {}))
    return ExperimentConfig(**values)
