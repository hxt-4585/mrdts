"""每次运行独立保存配置、代码版本和状态；训练器也可复用。"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import platform
import subprocess
import uuid

from experiments.config import PROJECT_ROOT
from methods.factory import COMPONENT_LABELS


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False),
                    encoding="utf-8")


def git_output(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_run(config, mode, scenes):
    components = config.method["components"]
    method_id = "_".join([config.method["solution"], *(COMPONENT_LABELS.get(components[name], components[name]) for name in
                                                     ("ordering", "flight", "scheduling"))])
    # Component names are factory-validated before using them in output paths.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    output = config.output_root / config.name / "runs" / method_id / f"seed_{config.seed}" / stamp
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"experiment": asdict(config), "scenes": scenes})
    metadata = dict(mode=mode, status="running", started_at=datetime.now(timezone.utc).isoformat(),
                    branch=git_output("branch", "--show-current"), commit=git_output("rev-parse", "HEAD"),
                    working_tree_status=git_output("status", "--short"), python=platform.python_version(),
                    seed=config.seed, method_id=method_id)
    write_json(output / "metadata.json", metadata)
    return output, metadata
