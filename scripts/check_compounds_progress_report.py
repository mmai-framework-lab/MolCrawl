"""Check a compounds batch-sweep progress report against the TSVs it quotes.

The report is prose, so a figure in it can drift from the run it describes -- a
stale paste, a hand recomputation, or a definition that moved between drafts.
Everything the report states about the grid is derivable from the two TSVs that
`compounds_batch_sweep_curves.py` writes, so this re-derives each figure and
asserts it appears in the right section of the report.

It checks presence, not formatting: a figure is expected somewhere inside the
section that owns it. A section is located by its heading, so a renamed heading
fails loudly rather than silently skipping its checks.

    python scripts/check_compounds_progress_report.py \
        --report <report>.md --summary rt1-...-summary-bert-<date>.tsv \
                             --curves  rt1-...-curves-bert-<date>.tsv

Exit status is 0 when every check passes and 1 otherwise, so it can gate a
report before it is sent.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

JUDGED = "eval_loss_mask"
BATCHES = (256, 512, 1024, 2048, 4096)
SIZES = ("small", "medium", "large", "xl")

# The detector's threshold, and the measured unigram cross-entropy at the [MASK]
# positions on this build (2026-11-02, mlm_lookup_baselines.py).
DETECTOR_THRESHOLD = 2.395

# A size's best arm is read at its lowest judged value. An arm whose end is far
# above its own minimum is not a measurement of the budget, so it is left out of
# the per-batch table -- the same rule as compounds_batch_sweep_spread.py.
CLIMB_MARGIN = 0.1

# Headings that own the figures. Each is matched as a prefix of a line.
SECTION_ALL = "## 1."
SECTION_COLLAPSE = "## 3."
SECTION_BEST = "### batch "
SECTION_EXCLUDED = "### 進行中"


class Checker:
    def __init__(self) -> None:
        self.bad: list[str] = []

    def __call__(self, label: str, want: object, got: object) -> None:
        ok = want == got
        print(f"{'ok  ' if ok else 'MISS'} {label}: expected={want!r} report={got!r}")
        if not ok:
            self.bad.append(label)

    def present(self, label: str, needle: str, haystack: str) -> None:
        self(label, True, needle in haystack)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with io.open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if not ln.startswith("#")]
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def section(text: str, heading: str, *, nxt: str | None = None) -> str:
    """The text under the heading that starts with `heading`, up to the next one."""
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.startswith(heading):
            start = i
            break
    if start is None:
        raise SystemExit(f"no section starting with {heading!r} in the report")
    depth = len(lines[start].split(" ", 1)[0])
    for j in range(start + 1, len(lines)):
        if nxt is not None:
            if lines[j].startswith(nxt):
                return "".join(lines[start:j])
            continue
        head = lines[j].split(" ", 1)[0]
        if head.startswith("#") and len(head) <= depth:
            return "".join(lines[start:j])
    return "".join(lines[start:])


def surviving(row: dict[str, str]) -> bool:
    if row["ended"] == "collapse-stop":
        return False
    if row["ended"] == "running":
        if float(row["final_value"]) - float(row["best_value"]) > CLIMB_MARGIN:
            return False
    return True


def check_counts(check: Checker, text: str, rows: list[dict[str, str]]) -> None:
    body = section(text, SECTION_ALL)
    check("arms in the summary", 48, len(rows))
    for state in ("budget", "collapse-stop", "running"):
        n = sum(1 for r in rows if r["ended"] == state)
        check.present(f"count {state} = {n}", str(n), body)
    for size in SIZES:
        n = sum(1 for r in rows if r["size"] == size and r["ended"] == "budget")
        if n:
            check.present(f"{size} at budget = {n}", str(n), body)


def check_best_table(check: Checker, text: str, rows: list[dict[str, str]]) -> None:
    body = section(text, SECTION_BEST)
    best: dict[tuple[str, int], float] = {}
    for row in rows:
        if not surviving(row):
            continue
        key = (row["size"], int(row["batch"]))
        value = float(row["best_value"])
        if key not in best or value < best[key]:
            best[key] = value
    for size in SIZES:
        values = []
        for batch in BATCHES:
            value = best.get((size, batch))
            if value is None:
                continue
            values.append(value)
            check.present(f"best {size} b{batch}", f"{value:.6f}", body)
        if len(values) == len(BATCHES):
            spread = max(values) - min(values)
            check.present(f"spread {size}", f"{spread:.6f}", body)


def check_collapsed(check: Checker, text: str, rows: list[dict[str, str]]) -> None:
    body = section(text, SECTION_COLLAPSE, nxt=SECTION_EXCLUDED)
    for row in rows:
        if row["ended"] != "collapse-stop":
            continue
        tag = f"{row['size']} b{row['batch']}"
        reached, planned = int(row["steps_reached"]), int(row["steps_planned"])
        check.present(f"collapse {tag} best", f"{float(row['best_value']):.6f}", body)
        check.present(f"collapse {tag} step", f"{reached:,}", body)
        check.present(f"collapse {tag} fraction", f"{100 * reached / planned:.1f} %", body)
        check.present(f"collapse {tag} stopped at",
                      f"{float(row['stopped_at_value']):.4f}", body)


def check_excluded_arm(
    check: Checker, text: str, rows: list[dict[str, str]], curves: Path
) -> None:
    """The arm the per-batch table leaves out gets a section of its own."""
    excluded = [r for r in rows if r["ended"] == "running" and not surviving(r)]
    if not excluded:
        print("-- no running arm is excluded; skipping its section")
        return
    if len(excluded) > 1:
        raise SystemExit(f"{len(excluded)} running arms excluded; the report assumes one")
    arm = excluded[0]
    body = section(text, SECTION_EXCLUDED)
    planned = int(arm["steps_planned"])

    series: list[tuple[int, float]] = []
    with io.open(curves, encoding="utf-8") as fh:
        head: list[str] | None = None
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if head is None:
                head = fields
                continue
            row = dict(zip(head, fields))
            if (row["size"], row["batch"], row["lr"], row["metric"]) == (
                arm["size"], arm["batch"], arm["lr"], JUDGED
            ):
                series.append((int(row["step"]), float(row["value"])))
    series.sort()
    check("excluded arm has a series", True, bool(series))
    check.present("excluded arm evaluation points", f"{len(series)} 点", body)
    check.present("excluded arm best", f"{float(arm['best_value']):.6f}", body)
    check.present("excluded arm final", f"{float(arm['final_value']):.6f}", body)
    delta = float(arm["final_value"]) - float(arm["best_value"])
    check.present("excluded arm final minus best", f"+{delta:.6f}", body)

    over = sum(1 for _, v in series if v >= DETECTOR_THRESHOLD)
    check("excluded arm points at the detector threshold", 0, over)
    check.present("excluded arm first value", f"{series[0][1]:.4f}", body)

    # Five-percent bands: the report tabulates n / min / max / mean per band.
    bands: dict[int, list[float]] = {}
    for step, value in series:
        bands.setdefault(int(100 * step / planned / 5), []).append(value)
    for band, values in sorted(bands.items()):
        tag = f"band {band * 5}-{(band + 1) * 5}%"
        check.present(f"{tag} n", f"| {len(values)} |", body)
        check.present(f"{tag} min", f"{min(values):.4f}", body)
        check.present(f"{tag} max", f"{max(values):.4f}", body)
        check.present(f"{tag} mean", f"{sum(values) / len(values):.4f}", body)

    # The arms the excluded one sits beside, which the section compares it to.
    for row in rows:
        if row["size"] != arm["size"] or row["batch"] != arm["batch"]:
            continue
        if row["lr"] == arm["lr"] or not surviving(row):
            continue
        check.present(f"sibling {row['lr']}", f"{float(row['best_value']):.6f}", body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--curves", required=True, type=Path)
    args = parser.parse_args()

    text = io.open(args.report, encoding="utf-8").read().lstrip("﻿")
    rows = read_tsv(args.summary)

    check = Checker()
    check_counts(check, text, rows)
    check_best_table(check, text, rows)
    check_collapsed(check, text, rows)
    check_excluded_arm(check, text, rows, args.curves)

    print()
    print(f"mismatches: {len(check.bad)}")
    for label in check.bad:
        print("  -", label)
    return 1 if check.bad else 0


if __name__ == "__main__":
    sys.exit(main())
