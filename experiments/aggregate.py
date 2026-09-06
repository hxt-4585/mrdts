"""收集已完成运行的结果；一行对应一次运行，不把重复种子冒充独立样本。"""

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.config import project_path


def aggregate(root):
    root = project_path(root)
    rows = []
    for path in sorted(root.rglob("summary.json")):
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "completed":
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(dict(method=metadata["method_id"], seed=metadata["seed"],
                         run=str(path.parent.relative_to(root)), **summary))
    if not rows:
        raise ValueError(f"No completed runs found in {root}")
    output = root / "analysis" / "comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Experiment result directory")
    args = parser.parse_args()
    print(aggregate(args.root))
