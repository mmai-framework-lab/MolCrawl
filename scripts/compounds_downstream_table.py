"""Collect the compounds downstream probe results into one table.

The trained score, the three untrained controls of the same architecture and
the fingerprint baseline are put on one row per arm so that the pretraining
effect is read against the spread of the untrained seeds rather than against
nothing. The improvement is signed by the metric's direction, so a positive
number always means pretraining helped.

No ordering across sizes is computed. The two tasks do not agree on one, and
the fold spread differs several-fold between arms, so a rank would read as a
finding that the measurement does not support.

Usage:
    compounds_downstream_table.py --trained DIR --untrained DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# A lower RMSE is better; a higher ROC-AUC is better.
DIRECTION = {"solubility": "lower", "herg": "higher"}

COLUMNS = [
    "task",
    "metric",
    "better",
    "arch",
    "size",
    "molecules",
    "trained",
    "trained_fold_sd",
    "untrained_seed0",
    "untrained_seed1",
    "untrained_seed2",
    "untrained_mean",
    "untrained_sd",
    "improvement_over_untrained",
    "fingerprint_baseline",
    "fingerprint_fold_sd",
    "adopted_checkpoint",
    "source",
]


def model_entries(path: Path) -> tuple[dict, dict]:
    with open(path) as fh:
        doc = json.load(fh)
    return doc, doc["models"]


def row(trained_path: Path, untrained_path: Path) -> list[str]:
    doc, trained_models = model_entries(trained_path)
    _, untrained_models = model_entries(untrained_path)

    task = doc["task"]
    baseline = trained_models["ECFP4+linear"]
    name = next(k for k in trained_models if k != "ECFP4+linear")
    trained = trained_models[name]

    seeds = []
    for seed in (0, 1, 2):
        key = f"{name}-untrained-seed{seed}"
        if key not in untrained_models:
            raise SystemExit(f"{untrained_path}: {key} is missing")
        seeds.append(untrained_models[key]["mean"])

    untrained_mean = statistics.fmean(seeds)
    gap = (
        untrained_mean - trained["mean"]
        if DIRECTION[task] == "lower"
        else trained["mean"] - untrained_mean
    )

    return [
        task,
        doc["metric"],
        DIRECTION[task],
        trained["arch"],
        trained["size"],
        str(doc["molecules"]),
        f"{trained['mean']:.4f}",
        f"{trained['sd']:.4f}",
        *[f"{v:.4f}" for v in seeds],
        f"{untrained_mean:.4f}",
        f"{statistics.stdev(seeds):.4f}",
        f"{gap:+.4f}",
        f"{baseline['mean']:.4f}",
        f"{baseline['sd']:.4f}",
        str(trained.get("adopted_checkpoint")),
        trained_path.name,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--untrained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    opts = parser.parse_args()

    rows = []
    for trained_path in sorted(opts.trained.glob("*.json")):
        if trained_path.name.startswith("smoke-"):
            continue
        untrained_path = opts.untrained / trained_path.name
        if not untrained_path.exists():
            raise SystemExit(f"no untrained control beside {trained_path.name}")
        rows.append(row(trained_path, untrained_path))

    # BOM so that the file is not taken for binary when it is shared on Slack.
    with open(opts.out, "w", encoding="utf-8-sig") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")

    for r in rows:
        print("\t".join(r))


if __name__ == "__main__":
    main()
