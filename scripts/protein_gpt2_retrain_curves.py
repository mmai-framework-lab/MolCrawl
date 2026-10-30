"""GPT-2 21-run retrain: per-run summary, loss curves, TSV.

boss protein-order-2026-09-24 §5. For each of the 21 runs, merge the logging_*.csv
(iter, train_loss, val_loss; a resume writes a new file) and emit:

  * a per-run summary: size, lr, seed, reached step, best val loss + its step
  * per-(size,lr) best val (seed spread where there are 3 seeds)
  * figure by size (LR overlaid) and by LR (size overlaid), val_loss vs iter
  * a TSV: size, lr, seed, iter, val_loss

The training-time val_loss is a subsample estimate; the ladder's canonical numbers are
the direct-read valid-100k values (score15/_direct), annotated at the final point where
available. No baseline line (CLM, not the BERT mask baseline). No GPU; compute node.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set from the command line: --runs holds the retrain2-gpt2-* run directories, --direct the
# direct-read valid-100k JSONs (score15/_direct), --out the report directory.
RUNS = None
DIRECT = None
OUT = None
SIZES = ["small", "medium", "large", "xl"]
LRS = ["0.75e-4", "1.5e-4", "3e-4", "6e-4"]
LR_COLOR = {"0.75e-4": "#7c3aed", "1.5e-4": "#2563eb", "3e-4": "#059669", "6e-4": "#dc2626"}
SIZE_COLOR = {"small": "#2563eb", "medium": "#059669", "large": "#d97706", "xl": "#dc2626"}


def parse_name(name):
    m = re.match(r"retrain2-gpt2-(\w+?)-lr([\d.]+e-\d)-s(\d+)$", name)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def load_run(run_dir):
    pts = {}
    for csv in sorted(glob.glob(os.path.join(run_dir, "logging_*.csv"))):
        with open(csv) as fh:
            next(fh, None)
            for line in fh:
                p = line.split(",")
                if len(p) < 3:
                    continue
                try:
                    it = int(p[0])
                    vl = float(p[2])
                except ValueError:
                    continue
                pts[it] = vl  # later segment overrides
    if not pts:
        return None
    items = sorted(pts.items())
    return {"iters": [i for i, _ in items], "val": [v for _, v in items]}


def direct_value(name):
    f = os.path.join(DIRECT, f"{name}.json")
    if os.path.exists(f):
        d = json.load(open(f))
        return list(d["results"].values())[0]["loss"]
    return None


def main():
    runs = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "retrain2-gpt2-*"))):
        key = parse_name(os.path.basename(d))
        if not key:
            continue
        r = load_run(d)
        if r:
            r["direct"] = direct_value(os.path.basename(d))
            runs[key] = r

    # ---- TSV ----
    tsv = Path(OUT) / "protein-gpt2-retrain-curves.tsv"
    with open(tsv, "w") as fh:
        fh.write("size\tlr\tseed\titer\tval_loss\n")
        for (size, lr, seed), r in runs.items():
            for it, v in zip(r["iters"], r["val"]):
                fh.write(f"{size}\t{lr}\t{seed}\t{it}\t{v:.6f}\n")
    print("wrote", tsv)

    # ---- summary ----
    print("\n=== §5-1 per-run: reached step, best val + step, direct valid100k ===")
    order = {s: i for i, s in enumerate(SIZES)}
    for (size, lr, seed), r in sorted(runs.items(), key=lambda kv: (order[kv[0][0]], kv[0][1], kv[0][2])):
        bi = min(range(len(r["val"])), key=lambda i: r["val"][i])
        dv = r["direct"]
        print(
            f"{size:6s} lr{lr:8s} s{seed}: reached {r['iters'][-1]:6d}  best_val {r['val'][bi]:.4f}@{r['iters'][bi]}"
            + (f"  valid100k {dv:.4f}" if dv is not None else "")
        )

    print("\n=== §5-2/3 per-(size,lr) best_val (max LR tried = 6e-4; nothing above) ===")
    agg = {}
    for (size, lr, _seed), r in runs.items():
        b = min(r["val"])
        agg.setdefault((size, lr), []).append(b)
    for size in SIZES:
        row = []
        for lr in LRS:
            vs = agg.get((size, lr))
            if vs:
                row.append(
                    f"{lr}:{min(vs):.4f}"
                    + (
                        f"(n={len(vs)},σ={(sum((x - sum(vs) / len(vs)) ** 2 for x in vs) / len(vs)) ** 0.5:.4f})"
                        if len(vs) > 1
                        else ""
                    )
                )
        print(f"  {size:6s}: " + "  ".join(row))

    # ---- figures ----
    def panel(ax, series, colormap, label_key):
        for lbl, r in series:
            ax.plot(r["iters"], r["val"], color=colormap[lbl], lw=1.2, label=label_key(lbl))
            bi = min(range(len(r["val"])), key=lambda i: r["val"][i])
            ax.plot(r["iters"][bi], r["val"][bi], "o", color=colormap[lbl], ms=5)
        ax.set_ylabel("val_loss (CLM)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    cap = (
        "protein GPT-2 21-run retrain | val_loss (training-time subsample) vs iter | "
        "max_iters 33,531 (9 epoch) | eff-batch 2,560 | LR swept 0.75e-4..6e-4 (6e-4 = top, nothing above)"
    )

    # by size: for each size, overlay its LRs (best seed per (size,lr) for clarity)
    figA, axes = plt.subplots(len(SIZES), 1, figsize=(9, 12), sharex=True)
    for ax, size in zip(axes, SIZES):
        series = []
        for lr in LRS:
            cand = [(k, r) for k, r in runs.items() if k[0] == size and k[1] == lr]
            if not cand:
                continue
            k, r = min(cand, key=lambda kr: min(kr[1]["val"]))  # best seed
            series.append((lr, r))
        panel(ax, series, LR_COLOR, lambda lr: f"lr {lr}")
        ax.set_title(f"protein GPT-2 {size}", fontsize=10, loc="left")
    axes[-1].set_xlabel("iter")
    figA.suptitle("protein GPT-2 retrain — by size (learning rates overlaid; best seed)", fontsize=11)
    figA.text(0.5, 0.005, cap, ha="center", fontsize=6.5)
    figA.tight_layout(rect=[0, 0.03, 1, 0.98])
    figA.savefig(Path(OUT) / "protein-gpt2-retrain-by-size.png", dpi=130)
    print("wrote", Path(OUT) / "protein-gpt2-retrain-by-size.png")

    # by LR: for each LR, overlay sizes (best seed)
    figB, axes = plt.subplots(len(LRS), 1, figsize=(9, 12), sharex=True)
    for ax, lr in zip(axes, LRS):
        series = []
        for size in SIZES:
            cand = [(k, r) for k, r in runs.items() if k[0] == size and k[1] == lr]
            if not cand:
                continue
            k, r = min(cand, key=lambda kr: min(kr[1]["val"]))
            series.append((size, r))
        panel(ax, series, SIZE_COLOR, lambda s: s)
        ax.set_title(f"protein GPT-2 lr {lr}", fontsize=10, loc="left")
    axes[-1].set_xlabel("iter")
    figB.suptitle("protein GPT-2 retrain — by learning rate (sizes overlaid; best seed)", fontsize=11)
    figB.text(0.5, 0.005, cap, ha="center", fontsize=6.5)
    figB.tight_layout(rect=[0, 0.03, 1, 0.98])
    figB.savefig(Path(OUT) / "protein-gpt2-retrain-by-lr.png", dpi=130)
    print("wrote", Path(OUT) / "protein-gpt2-retrain-by-lr.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, help="directory holding the retrain2-gpt2-* runs")
    ap.add_argument("--direct", required=True, help="directory of the direct-read valid-100k JSONs")
    ap.add_argument("--out", required=True, help="directory the TSV and figures are written to")
    a = ap.parse_args()
    RUNS, DIRECT, OUT = a.runs, a.direct, a.out
    main()
