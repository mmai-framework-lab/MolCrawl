"""Loss curves + TSV for the protein BERT 3x3 learning-rate grid.

boss protein-order-2026-09-24 §2. For each of the 9 runs, read the eval_loss_mask
trajectory from the newest checkpoint's trainer_state.json (HF restores the full
log_history on resume, so it covers all segments), and emit:

  * a TSV: size, lr, step, eval_loss_mask, segment
  * figure A: one panel per size (small/medium/large), the 3 learning rates overlaid
  * figure B: one panel per learning rate (1e-4/3e-4/1e-3), the 3 sizes overlaid

Both figures draw the unigram baseline 2.8947 as a horizontal line, mark each run's
best (lowest) eval_loss_mask with its step, and draw a vertical line at every segment
boundary (where a run was resumed after the 4-day limit). Collapsed arms are kept.

No GPU; run on a compute node. Reads only; writes to report/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set from the command line: --base is the bert-output directory holding the nine
# protein_sequence-<size>-lr<tag> runs, --out the report directory (outside any checkout).
BASE = None
OUT = None
BASELINE = 2.8947  # unigram marginal (no learning)
SIZES = ["small", "medium", "large"]
LRS = [("1e4", "1e-4"), ("3e4", "3e-4"), ("1e3", "1e-3")]
LR_COLOR = {"1e-4": "#2563eb", "3e-4": "#059669", "1e-3": "#dc2626"}
SIZE_COLOR = {"small": "#2563eb", "medium": "#059669", "large": "#dc2626"}


def newest_ckpt(run_dir):
    steps = [int(p.split("-")[-1]) for p in glob.glob(os.path.join(run_dir, "checkpoint-*"))]
    return max(steps) if steps else None


def segment_boundaries(run_dir):
    """Steps at which the run resumed (one per segment after the first).

    segments.log rows are `segment jobid ISO-timestamp`. Segment 2's start time T
    tells us it resumed from the newest checkpoint written before T; that checkpoint's
    step is the boundary the vline marks.
    """
    seg_path = os.path.join(run_dir, "segments.log")
    if not os.path.exists(seg_path):
        return []
    import datetime as dt

    starts = []
    for line in open(seg_path):
        parts = line.split()
        if len(parts) >= 3:
            try:
                starts.append(dt.datetime.fromisoformat(parts[2]).timestamp())
            except ValueError:
                pass
    bounds = []
    ckpts = [(int(p.split("-")[-1]), os.path.getmtime(p)) for p in glob.glob(os.path.join(run_dir, "checkpoint-*"))]
    for t in starts[1:]:  # skip the first segment (no boundary before it)
        before = [s for s, m in ckpts if m <= t]
        if before:
            bounds.append(max(before))
    return sorted(set(bounds))


def load_run(size, lrtag):
    run_dir = os.path.join(BASE, f"protein_sequence-{size}-lr{lrtag}")
    mx = newest_ckpt(run_dir)
    if mx is None:
        return None
    ts = json.load(open(os.path.join(run_dir, f"checkpoint-{mx}", "trainer_state.json")))
    pts = [(e["step"], e["eval_loss_mask"]) for e in ts.get("log_history", []) if "eval_loss_mask" in e]
    pts.sort()
    return {"steps": [p[0] for p in pts], "loss": [p[1] for p in pts], "bounds": segment_boundaries(run_dir)}


def main():
    runs = {}
    for size in SIZES:
        for lrtag, lrlabel in LRS:
            r = load_run(size, lrtag)
            if r:
                runs[(size, lrlabel)] = r

    # ---- TSV ----
    tsv = Path(OUT) / "protein-bert-grid-curves.tsv"
    with open(tsv, "w") as fh:
        fh.write("size\tlr\tstep\teval_loss_mask\tsegment\n")
        for (size, lr), r in runs.items():
            bounds = r["bounds"]
            for s, v in zip(r["steps"], r["loss"]):
                seg = 1 + sum(1 for b in bounds if s > b)
                fh.write(f"{size}\t{lr}\t{s}\t{v:.6f}\t{seg}\n")
    print("wrote", tsv)

    cap = (
        "protein BERT 3x3 LR grid | eval_loss_mask on a fixed 10,000-row valid subset "
        "(seed 42), every 100 steps | max_steps 33,531 (9 epoch) | eff-batch 2,560 (8x80x4) "
        "| baseline 2.8947 = unigram"
    )

    # ---- figure A: per size, LR overlaid ----
    figA, axes = plt.subplots(len(SIZES), 1, figsize=(9, 10), sharex=True)
    for ax, size in zip(axes, SIZES):
        for _lrtag, lr in LRS:
            r = runs.get((size, lr))
            if not r:
                continue
            ax.plot(r["steps"], r["loss"], color=LR_COLOR[lr], lw=1.2, label=f"lr {lr}")
            bi = min(range(len(r["loss"])), key=lambda i: r["loss"][i])
            ax.plot(r["steps"][bi], r["loss"][bi], "o", color=LR_COLOR[lr], ms=5)
            ax.annotate(
                f"{r['loss'][bi]:.3f}@{r['steps'][bi]}",
                (r["steps"][bi], r["loss"][bi]),
                fontsize=7,
                color=LR_COLOR[lr],
                xytext=(0, 5),
                textcoords="offset points",
            )
            for b in r["bounds"]:
                ax.axvline(b, color=LR_COLOR[lr], ls=":", lw=0.7, alpha=0.5)
        ax.axhline(BASELINE, color="black", ls="--", lw=0.8)
        ax.text(0, BASELINE, " 2.8947 baseline", fontsize=7, va="bottom")
        ax.set_title(f"protein BERT {size}", fontsize=10, loc="left")
        ax.set_ylabel("eval_loss_mask")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("step")
    figA.suptitle("protein BERT LR grid — by size (learning rates overlaid)", fontsize=11)
    figA.text(0.5, 0.005, cap, ha="center", fontsize=6.5, wrap=True)
    figA.tight_layout(rect=[0, 0.03, 1, 0.98])
    pA = Path(OUT) / "protein-bert-grid-by-size.png"
    figA.savefig(pA, dpi=130)
    print("wrote", pA)

    # ---- figure B: per LR, size overlaid ----
    figB, axes = plt.subplots(len(LRS), 1, figsize=(9, 10), sharex=True)
    for ax, (_lrtag, lr) in zip(axes, LRS):
        for size in SIZES:
            r = runs.get((size, lr))
            if not r:
                continue
            ax.plot(r["steps"], r["loss"], color=SIZE_COLOR[size], lw=1.2, label=size)
            bi = min(range(len(r["loss"])), key=lambda i: r["loss"][i])
            ax.plot(r["steps"][bi], r["loss"][bi], "o", color=SIZE_COLOR[size], ms=5)
            ax.annotate(
                f"{r['loss'][bi]:.3f}@{r['steps'][bi]}",
                (r["steps"][bi], r["loss"][bi]),
                fontsize=7,
                color=SIZE_COLOR[size],
                xytext=(0, 5),
                textcoords="offset points",
            )
            for b in r["bounds"]:
                ax.axvline(b, color=SIZE_COLOR[size], ls=":", lw=0.7, alpha=0.5)
        ax.axhline(BASELINE, color="black", ls="--", lw=0.8)
        ax.text(0, BASELINE, " 2.8947 baseline", fontsize=7, va="bottom")
        ax.set_title(f"protein BERT lr {lr}", fontsize=10, loc="left")
        ax.set_ylabel("eval_loss_mask")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("step")
    figB.suptitle("protein BERT LR grid — by learning rate (sizes overlaid)", fontsize=11)
    figB.text(0.5, 0.005, cap, ha="center", fontsize=6.5, wrap=True)
    figB.tight_layout(rect=[0, 0.03, 1, 0.98])
    pB = Path(OUT) / "protein-bert-grid-by-lr.png"
    figB.savefig(pB, dpi=130)
    print("wrote", pB)

    # summary to stdout
    print("\n=== per-run min eval_loss_mask (step) ===")
    for (size, lr), r in sorted(runs.items()):
        bi = min(range(len(r["loss"])), key=lambda i: r["loss"][i])
        last = r["loss"][-1]
        flag = " COLLAPSED(>=baseline)" if last >= BASELINE else ""
        print(
            f"{size:6s} lr{lr}: min {r['loss'][bi]:.4f}@{r['steps'][bi]}  last {last:.4f}@{r['steps'][-1]}  seg_bounds={r['bounds']}{flag}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="bert-output directory holding the 9 grid runs")
    ap.add_argument("--out", required=True, help="directory the TSV and figures are written to")
    a = ap.parse_args()
    BASE, OUT = a.base, a.out
    main()
