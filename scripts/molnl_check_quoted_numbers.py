#!/usr/bin/env python3
"""Check numbers quoted back at us against the runs they came from.

Orders and reports quote values from earlier reports, and a value that was right once is
not right after the run it describes has moved. This recomputes the ones the 2026-09-30
order rests on, straight from each run's trainer_state.json, and prints them beside what
was quoted so a mismatch is visible rather than inherited.

    python scripts/molnl_check_quoted_numbers.py --runs-root <bert-output-shuffled>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st

# The 12,000-step arms and the window means the order gives for them (§2.1).
QUOTED_WINDOWS = {
    "bert_small_lr1e3": [0.1102, 0.0930, 0.0806],
    "bert_medium_lr3e4": [0.1006, 0.0834, 0.0733],
    "bert_large_lr1e4": [0.1543, 0.1345, 0.1232],
}
RANGES = ((6000, 8000), (8000, 10000), (10000, 12000))
# Non-embedding parameter counts, measured from the checkpoints (job 147702).
PARAMS = {"small": 85698392, "medium": 303413336, "large": 575231192}
# The rate that held and the rate that diverged, per size (§5).
BANDS = {"small": (1e-3, 1.3e-3), "medium": (4.1e-4, 5.5e-4), "large": (2.3e-4, 3e-4)}


def series(runs_root, run):
    cks = sorted(glob.glob(os.path.join(runs_root, run, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        return []
    history = json.load(open(os.path.join(cks[-1], "trainer_state.json")))["log_history"]
    return sorted((e["step"], e["eval_loss_mask"]) for e in history if "eval_loss_mask" in e)


def slope(points):
    xs = [math.log10(x) for x, _ in points]
    ys = [math.log10(y) for _, y in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--tolerance", type=float, default=0.0005)
    a = ap.parse_args()

    print("=== 12,000-step arms, mean over each 2,000-step window ===")
    problems = 0
    for run, quoted in QUOTED_WINDOWS.items():
        pts = series(a.runs_root, run)
        if not pts:
            print(f"  {run}: no series")
            problems += 1
            continue
        got = [st.mean([v for s, v in pts if lo < s <= hi]) for lo, hi in RANGES]
        ok = all(abs(g - q) < a.tolerance for g, q in zip(got, quoted))
        problems += 0 if ok else 1
        print(f"  {run:<20} measured {got[0]:.4f} {got[1]:.4f} {got[2]:.4f}"
              f"   quoted {quoted[0]} {quoted[1]} {quoted[2]}   {'ok' if ok else 'MISMATCH'}")
        low_step, low = min(pts, key=lambda p: p[1])
        print(f"       last 2,000 steps moved {got[2] - got[1]:+.4f}; "
              f"lowest point {low:.4f} at step {low_step:,} of {pts[-1][0]:,}")

    print("\n=== divergence boundary against size ===")
    mids = {k: math.sqrt(lo * hi) for k, (lo, hi) in BANDS.items()}
    for size, mid in mids.items():
        print(f"  {size:<7} held {BANDS[size][0]:g}, diverged {BANDS[size][1]:g}, "
              f"geometric middle {mid:.3e}  (N = {PARAMS[size]:,})")
    a_fit = slope([(PARAMS[k], mids[k]) for k in PARAMS])
    print(f"  least-squares exponent {a_fit:+.3f}   (the order says N^-0.84)")
    for lo, hi in (("small", "medium"), ("medium", "large")):
        pair = math.log10(mids[hi] / mids[lo]) / math.log10(PARAMS[hi] / PARAMS[lo])
        print(f"  {lo} to {hi}: {pair:+.3f}")

    # What the order's three xl rates assume, and what the same exponent gives.
    n_xl = 12 * 48 * 1600 ** 2          # main.py's xl: 48 layers, hidden 1,600
    print(f"\n=== xl ===\n  12 x L x d^2 gives {n_xl:,} ({n_xl / 1e6:.0f}M), "
          f"{n_xl / PARAMS['large']:.2f}x large")
    for ratio, label in ((2.4, "the order's assumed 2.4x"),
                         (n_xl / PARAMS["large"], "the estimate above")):
        print(f"  {label}: boundary near {mids['large'] * ratio ** a_fit:.3e} "
              f"(the order centres its three rates on 1.2e-4)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
