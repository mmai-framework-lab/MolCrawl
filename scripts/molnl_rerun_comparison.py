#!/usr/bin/env python3
"""Does repeating a run move its number by more than the seeds already do?

The 2026-10-14 order asks for every size to be run again with the validation set held
fixed, and the question behind it is whether the per-size picture survives. Comparing the
best value of two runs answers a different question than it looks like: the best is the
minimum of however many evaluations a run happened to log, so a longer or more frequently
evaluated run wins on luck. This reads the mean of the last ten evaluations instead, and
puts the difference next to the spread the seeds of the same arm already show.

    python scripts/molnl_rerun_comparison.py --table <all-curves tsv>
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

WINDOW = 10  # evaluations averaged at the end of a run

PAIRS = [
    ("bert", "small", "bert_small_24k_lr1e3", "bert_small_24k_rerun"),
    ("bert", "medium", "bert_medium_24k_lr3e4", "bert_medium_24k_rerun"),
    ("bert", "large", "bert_large_24k_lr1p7e4", "bert_large_24k_rerun"),
    ("gpt2", "small", "gpt2_small_1500_lr1p2e3", "gpt2_small_1500_fixedval"),
    ("gpt2", "medium", "gpt2_medium_1500_lr6e4", "gpt2_medium_1500_fixedval"),
    ("gpt2", "large", "gpt2_large_1500_lr3e4", "gpt2_large_1500_fixedval"),
    ("gpt2", "xl", "gpt2_xl_1500_lr3e4", "gpt2_xl_1500_fixedval"),
]


def load(path):
    series = defaultdict(dict)
    for row in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        if row["split"] == "validation" and row["metric"] != "val_loss_full_split":
            series[row["run"]][int(row["step"])] = float(row["value"])
    return series


def window_mean(points):
    steps = sorted(points)[-WINDOW:]
    return sum(points[s] for s in steps) / len(steps), steps[0], steps[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True)
    a = ap.parse_args()
    series = load(a.table)

    print(f"  最後の {WINDOW} 評価の平均で比べる。括弧内はその区間の step。\n")
    header = (f"{'arch':<5}{'size':<8}{'元':>10}{'再走行':>10}{'差':>9}"
              f"{'seed 幅':>9}{'差/幅':>8}  区間")
    print(header)
    print("  " + "-" * (len(header) + 10))
    for arch, size, original, rerun in PAIRS:
        if original not in series or rerun not in series:
            print(f"{arch:<5}{size:<8}  (表に無い: "
                  f"{original if original not in series else rerun})")
            continue
        a_mean, _, _ = window_mean(series[original])
        b_mean, lo, hi = window_mean(series[rerun])
        seeds = [window_mean(series[n])[0] for n in
                 (original, original + "_seed1", original + "_seed17") if n in series]
        spread = (max(seeds) - min(seeds)) if len(seeds) >= 2 else None
        ratio = (abs(b_mean - a_mean) / spread) if spread else None
        print(f"{arch:<5}{size:<8}{a_mean:>10.4f}{b_mean:>10.4f}{b_mean - a_mean:>+9.4f}"
              f"{(f'{spread:.4f}' if spread else '—'):>9}"
              f"{(f'{ratio:.2f}x' if ratio else '—'):>8}  {lo}-{hi}"
              + (f"  (seed {len(seeds)} 本)" if spread else "  (seed 1 本)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
