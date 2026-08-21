"""Regenerate the genome GPT-2 21-subset val-loss figure from the SLURM logs.

Run from a checkout that can see workflows/slurm-logs/ and
learning_source_genome_runs/. The PNG is a build artefact and is not committed;
this script and the CSVs under learning_source_genome_runs/_results/ are what
the figure is reproduced from.

Only runs whose runtime output reports 2,621,440 tokens per iteration are used.
The configs echo an older 655,360 figure in a comment, and the logs carry both
strings, so matching on the runtime line is what keeps the pre-fix 640-batch
runs out of the figure.

Colours are the reference categorical palette, slots 1-3, used unchanged.
"""

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#bcbbb4"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

PRODUCTION_TOKENS = "tokens per iteration will be: 2,621,440"
EVAL_RE = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")


def curve(sub):
    """Return the val-loss series of the newest production run for one subset."""
    for path in sorted(glob.glob(f"workflows/slurm-logs/mc-gen-gpt2-{sub}-*.out"), key=os.path.getmtime, reverse=True):
        text = open(path, errors="ignore").read()
        if PRODUCTION_TOKENS not in text:
            continue
        return [(int(s), float(v)) for s, _, v in EVAL_RE.findall(text)]
    return []


subs = [os.path.basename(d).replace("gpt2-small-", "") for d in sorted(glob.glob("learning_source_genome_runs/gpt2-small-*"))]
data = {s: curve(s) for s in subs}
data = {k: v for k, v in data.items() if v}

HL = {"global_random_seed3": S1, "eukaryote_matched_random_seed4": S2, "mammal_centered": S3}
LBL = {
    "global_random_seed3": "global_random_seed3",
    "eukaryote_matched_random_seed4": "eukaryote_…_seed4",
    "mammal_centered": "mammal_centered",
}

# For each run: the step of its largest spike, that spike's size, and where it ended.
stat = {}
for s, ev in data.items():
    running_min = float("inf")
    rise = []
    for st, v in ev:
        running_min = min(running_min, v)
        rise.append((st, v - running_min))
    peak = max(rise, key=lambda r: r[1])
    stat[s] = (peak[0], peak[1], rise[-1][1])

fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e8e7e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d5d4cc")
    ax.tick_params(colors=INK2, labelsize=9)

# (a) val curves, zoomed past the initial drop
ax = axes[0]
for s, ev in data.items():
    if s in HL:
        continue
    ax.plot([p[0] for p in ev], [p[1] for p in ev], color=MUTED, lw=1.0, alpha=0.8, zorder=1)
for s, c in HL.items():
    ev = data[s]
    ax.plot([p[0] for p in ev], [p[1] for p in ev], color=c, lw=2.0, zorder=3, label=LBL[s])
    bx, by = min(ev, key=lambda r: r[1])
    ax.plot([bx], [by], "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=4)
    ax.annotate(
        f"best @ {bx // 1000}k",
        (bx, by),
        textcoords="offset points",
        xytext=(-4, -17),
        fontsize=8.5,
        color=c,
        ha="center",
        fontweight="bold",
    )
ax.set_xlim(0, 56000)
ax.set_ylim(1.06, 1.40)
ax.set_title("(a) val loss, plateau region", fontsize=11.5, color=INK, loc="left", pad=9)
ax.set_xlabel("optimizer step", fontsize=9.5, color=INK2)
ax.set_ylabel("val loss", fontsize=9.5, color=INK2)
ax.text(
    0.985,
    0.965,
    "grey = the other 18 subsets\nlevels differ by subset — different data",
    transform=ax.transAxes,
    fontsize=8.5,
    color=INK2,
    va="top",
    ha="right",
)
ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")

# (b) spike timing vs how the run ended
ax = axes[1]
grp = [
    (">0.05 at the end", S1, lambda f: f > 0.05),
    ("0.01–0.05", S2, lambda f: 0.01 < f <= 0.05),
    ("≤0.01", S3, lambda f: f <= 0.01),
]
for lbl, c, test in grp:
    xs = [stat[s][0] for s in stat if test(stat[s][2])]
    ys = [stat[s][1] for s in stat if test(stat[s][2])]
    ax.scatter(xs, ys, s=95, color=c, edgecolor=SURFACE, linewidth=1.6, zorder=3, label=lbl)
ann = {"global_random_seed6": ((14, 10), "left"), "eukaryote_matched_random_seed4": ((12, -6), "left")}
for s, (off, ha) in ann.items():
    x, y, f = stat[s]
    ax.annotate(
        f"{s}\nspike {y:.3f} @ {x // 1000}k  →  ended {f:+.4f}",
        (x, y),
        textcoords="offset points",
        xytext=off,
        fontsize=8.2,
        color=INK2,
        ha=ha,
        bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=1.5),
    )
ax.set_xlim(0, 72000)
ax.set_title("(b) later spikes tend to leave a larger reversal", fontsize=11.5, color=INK, loc="left", pad=9)
ax.set_xlabel("step of the run's largest spike", fontsize=9.5, color=INK2)
ax.set_ylabel("size of that spike (val − running min)", fontsize=9.5, color=INK2)
ax.text(
    0.015,
    0.03,
    "18 of 21 runs spike by ≥0.02 at some point.\n"
    "Rank correlation with the final reversal: spike step +0.515, steps left −0.443, spike size +0.218 (n=21).\n"
    "Caveat: one repeat of identical settings gave reversals 0.0013 and 0.0760, so gaps of that size are not\n"
    "separable from run-to-run variation. Read this panel as a tendency, not a mechanism.",
    transform=ax.transAxes,
    fontsize=8.2,
    color=INK2,
    va="bottom",
)
ax.legend(
    frameon=False,
    fontsize=9,
    labelcolor=INK,
    loc="lower right",
    title="reversal at final step",
    title_fontsize=8.5,
)

fig.suptitle(
    "genome GPT-2 at peak LR 1e-4 — spikes are common and usually recoverable; the best-val checkpoint is what is adopted",
    fontsize=13,
    color=INK,
    x=0.008,
    ha="left",
    y=0.98,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("tmp/figs/b21-val-curves.png", dpi=170, facecolor=SURFACE)
print("saved tmp/figs/b21-val-curves.png")
