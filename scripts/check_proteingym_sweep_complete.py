"""Count whether a ProteinGym sweep actually scored every assay.

protein-judgment-2026-11-09 §3.1. The sweep resumes by skipping assays that already
look done. The reliable "done" marker is a parseable metrics.json with the score keys
present (a NaN spearman still counts -- the assay was scored, the value is just
undefined), not the mere existence of a file in the leaf: a segment cut by the wall can
leave a half-written REPORT.md and that assay would otherwise be skipped unscored.

This walks one or more eval slugs and reports, per slug, how many of the 217 assays are
complete, how many are NaN, and names anything missing or corrupt so it can be rescored.

Run:  python scripts/check_proteingym_sweep_complete.py [slug ...]
With no slug it scans every ``protein_sequence-*-s42`` slug under the eval root.
Exit status is non-zero if any assay is missing or corrupt.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "learning_source_20260730_protein_uniref50/experiment_data/eval"
PGDIR = REPO / "learning_source_20260730_protein_uniref50/eval/proteingym/unpacked/DMS_ProteinGym_substitutions"


def classify(metrics_path: Path) -> str:
    """ok (scored, defined), nan (scored, undefined), corrupt, or missing."""
    if not metrics_path.exists():
        return "missing"
    try:
        d = json.load(open(metrics_path))
    except Exception:
        return "corrupt"
    m = d.get("metrics", {})
    if "spearman" not in m or d.get("details", {}).get("num_variants") is None:
        return "corrupt"
    s = m["spearman"]
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "nan"
    return "ok"


def main() -> None:
    assays = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(PGDIR / "*.csv")))
    slugs = sys.argv[1:]
    if not slugs:
        slugs = sorted(os.path.basename(p) for p in glob.glob(str(EVAL / "protein_sequence-*-s42")))
    if not slugs:
        sys.exit("no slugs given and none matched protein_sequence-*-s42")

    bad = 0
    for slug in slugs:
        counts = {"ok": 0, "nan": 0, "corrupt": 0, "missing": 0}
        problems = []
        for a in assays:
            c = classify(EVAL / slug / f"proteingym_{a}_sweep" / "metrics.json")
            counts[c] += 1
            if c in ("corrupt", "missing"):
                problems.append(f"{c}: {a}")
        total = len(assays)
        print(f"{slug}: scored={counts['ok'] + counts['nan']}/{total} "
              f"(ok={counts['ok']}, nan={counts['nan']}, corrupt={counts['corrupt']}, missing={counts['missing']})")
        for p in problems:
            print(f"    {p}")
        bad += counts["corrupt"] + counts["missing"]

    print(f"\nassays per slug: {len(assays)}; slugs checked: {len(slugs)}; need-rescore: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
