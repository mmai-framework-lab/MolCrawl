"""Check the premises the rna batch-size directive of 2026-11-20 states, by measuring.

The directive prints numbers it reasoned from, and a grid of 68 arms is built on top
of them. Four of those numbers are measurements of the existing grid, so they can be
checked rather than taken:

  §2  the train split is 34,407,040 packed blocks, and the budget of 309,657,600
      blocks is 9.00 epochs of it.
  §3  "medium best 3.0e-4, +0.020 at 4.1e-4 (1.37x), collapses at 5.5e-4 (1.83x)"
      and the same shape for large -- the claim that 1.4x stays short of the cliff,
      which is why the three-rate batches put an arm there.
  §10 the number of blocks each adopted arm spent above an eval_loss_mask of 8.0
      (small 4,608,000 / medium 3,840,000 / large 6,144,000 / xl 151,040,000).

and one number the directive does not state but the grid needs: where a healthy arm
crosses the collapse detector's threshold. The detector stops a run whose judged
metric sits at or above a floor for N evaluations, and `degenerate_grace_steps` is
compared against the absolute step count, so a grid whose step counts differ by 16x
between batches cannot share one grace value. The column position of the crossing is
the same for every batch; the step is not.

Reads the curve TSV and the grid summary; writes a TSV and prints what disagrees.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

SEQUENCE_LENGTH = 1024
GRID_BATCH = 2560  # the existing grid's global batch, in blocks per step

#: §2.
STATED_TRAIN_BLOCKS = 34_407_040
STATED_BUDGET_BLOCKS = 309_657_600
STATED_EPOCHS = 9.00

#: §3. (size, cited lr, the claim about it)
STATED_RATE_CLAIMS = [
    ("medium", 3.0e-4, "best", None),
    ("medium", 4.1e-4, "worse_by", 0.020),
    ("medium", 5.5e-4, "collapses", None),
    ("large", 1.7e-4, "best", None),
    ("large", 2.3e-4, "worse_by", 0.0015),
    ("large", 3.0e-4, "collapses", None),
]

#: §10, in blocks.
STATED_ABOVE_8 = {"small": 4_608_000, "medium": 3_840_000,
                  "large": 6_144_000, "xl": 151_040_000}

#: The arms §1 of the extension directive adopted, which this grid scales from.
ADOPTED = {"small": 3.0e-4, "medium": 3.0e-4, "large": 1.7e-4, "xl": 2.5e-5}

#: The collapse detector's threshold, measured for rna on 2026-11-18 and approved for
#: the four extension arms. Checked here against the healthy arms, not assumed.
THRESHOLDS = (9.0, 8.5, 8.0)


def load_curves(path, metric):
    """{(size, lr): [(step, value)]} for the 9-epoch BERT arms, in step order."""
    series = defaultdict(list)
    metrics = set()
    with open(path, encoding="utf-8") as handle:
        rows = csv.reader((ln for ln in handle if not ln.startswith("#")), delimiter="\t")
        header = next(rows)
        idx = {name: i for i, name in enumerate(header)}
        for row in rows:
            if not row or row[idx["arch"]] != "bert":
                continue
            metrics.add(row[idx["metric"]])
            if row[idx["metric"]] != metric or row[idx["split"]] != "val":
                continue
            key = (row[idx["size"]], float(row[idx["lr"]]))
            series[key].append((int(row[idx["step"]]), float(row[idx["value"]])))
    for key in series:
        series[key].sort()
    return series, metrics


def load_summary(path):
    out = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader((ln for ln in handle if not ln.startswith("#")),
                                  delimiter="\t"):
            if row.get("arch") not in (None, "", "bert") and row.get("arch") != "bert":
                continue
            out[(row["size"], float(row["lr"]))] = row
    return out


def first_below(points, threshold):
    """The first (step, value) whose value is below ``threshold``, or None."""
    for step, value in points:
        if value < threshold:
            return step, value
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curves", required=True, help="rt1 rna curve TSV")
    ap.add_argument("--summary", required=True, help="rna BERT grid summary TSV")
    ap.add_argument("--train-bin", required=True, help="the train split's .bin")
    ap.add_argument("--metric", default="eval_loss_mask")
    ap.add_argument("--out", help="TSV to write")
    args = ap.parse_args()

    disagreements = []

    print("== §2 the budget")
    size_bytes = os.path.getsize(args.train_bin)
    blocks = size_bytes // 2 // SEQUENCE_LENGTH
    print(f"   train split {blocks:,} blocks  (stated {STATED_TRAIN_BLOCKS:,})")
    if blocks != STATED_TRAIN_BLOCKS:
        disagreements.append(f"train split is {blocks:,}, stated {STATED_TRAIN_BLOCKS:,}")
    epochs = STATED_BUDGET_BLOCKS / blocks
    short_by = blocks * 9 - STATED_BUDGET_BLOCKS
    print(f"   budget {STATED_BUDGET_BLOCKS:,} blocks = {epochs:.6f} epochs"
          f"  ({'short by' if short_by > 0 else 'over by'} {abs(short_by):,} blocks"
          f" against a whole 9)")
    if abs(epochs - STATED_EPOCHS) > 0.0005:
        disagreements.append(f"budget is {epochs:.6f} epochs, stated {STATED_EPOCHS}")
    print(f"   tokens {STATED_BUDGET_BLOCKS * SEQUENCE_LENGTH / 1e9:.2f} G")

    series, metrics = load_curves(args.curves, args.metric)
    if not series:
        print(f"   no {args.metric} series; metrics present: {sorted(metrics)}")
        return 2
    summary = load_summary(args.summary)

    print(f"\n== §3 the rate claims ({args.metric})")
    for size, rate, kind, stated in STATED_RATE_CLAIMS:
        points = series.get((size, rate))
        row = summary.get((size, rate))
        if points is None or row is None:
            print(f"   {size} {rate:g}: no arm")
            disagreements.append(f"{size} {rate:g} has no arm")
            continue
        best = min(v for _, v in points)
        collapsed = row.get("collapsed") == "1"
        pattern = row.get("pattern", "")
        turned = row.get("turned_step", "")
        note = f"best {best:.4f}  collapsed={int(collapsed)}  {pattern}"
        if turned:
            note += f" @ {int(turned):,}"
        if kind == "worse_by":
            adopted = series.get((size, ADOPTED[size]))
            delta = best - min(v for _, v in adopted)
            print(f"   {size} {rate:g}: {note}  (+{delta:.4f} on the adopted arm,"
                  f" stated +{stated})")
            if abs(delta - stated) > 0.0006:
                disagreements.append(
                    f"{size} {rate:g} is +{delta:.4f}, stated +{stated}")
            if collapsed:
                disagreements.append(
                    f"{size} {rate:g} is cited as a healthy cost of +{stated} but the"
                    f" arm collapsed ({pattern}"
                    f"{' @ ' + format(int(turned), ',') if turned else ''})")
        else:
            print(f"   {size} {rate:g}: {note}  (stated: {kind})")
            if kind == "collapses" and not collapsed:
                disagreements.append(f"{size} {rate:g} did not collapse")
            if kind == "best" and collapsed:
                disagreements.append(f"{size} {rate:g} is the adopted arm but collapsed")

    print("\n== §10 blocks above a mask loss of 8.0, and where a healthy arm crosses")
    out_rows = []
    for size, rate in sorted(ADOPTED.items(), key=lambda kv: ["small", "medium", "large",
                                                              "xl"].index(kv[0])):
        points = series[(size, rate)]
        line = [f"   {size} {rate:g}:"]
        for threshold in THRESHOLDS:
            hit = first_below(points, threshold)
            if hit is None:
                line.append(f"<{threshold}: never")
                continue
            step, value = hit
            cols = step * GRID_BATCH
            line.append(f"<{threshold}: step {step:,} = {cols:,} blocks"
                        f" ({100 * cols / STATED_BUDGET_BLOCKS:.2f}% of budget)")
            out_rows.append({"size": size, "lr": f"{rate:g}", "threshold": threshold,
                             "step": step, "blocks": cols, "value": f"{value:.4f}",
                             "share_of_budget": f"{cols / STATED_BUDGET_BLOCKS:.6f}"})
            if threshold == 8.0:
                stated = STATED_ABOVE_8[size]
                print("   " + "\n      ".join(line))
                line = [f"      stated {stated:,} blocks"
                        f" -> {'matches' if stated == cols else 'differs'}"]
                if stated != cols:
                    disagreements.append(
                        f"{size} crosses 8.0 at {cols:,} blocks, stated {stated:,}")
        print("      " + "\n      ".join(line))

    if args.out and out_rows:
        with open(args.out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nwrote {args.out}")

    print(f"\n== 食い違い {len(disagreements)} 件")
    for item in disagreements:
        print(f"   - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
