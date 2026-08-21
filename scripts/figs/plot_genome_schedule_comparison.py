"""Regenerate the genome GPT-2 schedule-comparison figure from the SLURM logs.

Run from a checkout that can see workflows/slurm-logs/. The PNG is a build
artefact and is not committed; this script and the CSVs under
learning_source_genome_runs/_results/ are what the figure is reproduced from.

All five series are the same subset (mammal_centered). Four are the stability
sweep of 2026-08-14; the fifth is the 2026-08-17 production run, which repeats
the sweep's B settings exactly (peak 1e-4, decay 54,634, global batch 2,560,
seed 45, from scratch, no resume). It is plotted so the figure carries its own
evidence of how much a run varies when nothing is changed.

Colours are the reference categorical palette, slots 1-4, used unchanged. The
repeat shares B's colour and is dashed, because it is the same condition rather
than a new one.
"""

import glob
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

EVAL_RE = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")


def curve(pat):
    """Return the val-loss series from the first log matching `pat`."""
    path = sorted(glob.glob(pat))[0]
    text = open(path, errors="ignore").read()
    return [(int(s), float(v)) for s, _, v in EVAL_RE.findall(text)]


runs = [
    ("base", "peak 6e-4, decay 54,634", S2, "-", curve("workflows/slurm-logs/mc-gen-stab-base-25495.out")),
    ("D", "peak 6e-4, decay 13,658", S4, "-", curve("workflows/slurm-logs/mc-gen-stab-D-decay025-25499.out")),
    ("B", "peak 1e-4, decay 54,634  (sweep, 08-14)", S1, "-", curve("workflows/slurm-logs/mc-gen-stab-B-lr1e4-25497.out")),
    ("B repeat", "same settings  (production, 08-17)", S1, "--", curve("workflows/slurm-logs/mc-gen-gpt2-mammal_centered-28813.out")),
    ("(c)", "peak 6e-4, decay 1,802", S3, "-", curve("workflows/slurm-logs/mc-gen-stab2-H-6e4to1e4-*.out")),
]

fig, (axT, axB) = plt.subplots(
    2,
    1,
    figsize=(10.8, 8.8),
    facecolor=SURFACE,
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.13},
)
for ax in (axT, axB):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e8e7e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d5d4cc")
    ax.tick_params(colors=INK2, labelsize=9)

for tag, desc, c, style, ev in runs:
    xs = [p[0] for p in ev]
    ys = [p[1] for p in ev]
    best = min(ev, key=lambda r: r[1])
    for ax in (axT, axB):
        ax.plot(xs, ys, color=c, lw=2.0, ls=style, zorder=3, label=f"{tag:<9} {desc}" if ax is axT else None)
        ax.plot([best[0]], [best[1]], "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=4)

axT.set_ylim(1.05, 1.97)
axT.set_xlim(0, 60000)
axT.set_title("Full range — base swings up to 1.91", fontsize=11, color=INK, loc="left", pad=8)
axT.set_ylabel("val loss", fontsize=9.5, color=INK2)
axT.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right", ncol=2, columnspacing=1.6)
axT.annotate(
    "base — final 1.6853",
    (6000, 1.79),
    fontsize=9.5,
    color=S2,
    fontweight="bold",
    bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=2),
)

axB.set_ylim(1.072, 1.225)
axB.set_title("Same data, zoomed to where the other four separate", fontsize=11, color=INK, loc="left", pad=8)
axB.set_xlabel("optimizer step", fontsize=9.5, color=INK2)
axB.set_ylabel("val loss", fontsize=9.5, color=INK2)
for tag, _desc, c, _style, ev in runs:
    if tag == "base":
        continue
    xs = [p[0] for p in ev]
    ys = [p[1] for p in ev]
    dy = {"D": 0, "B": 10, "B repeat": 0, "(c)": -10}[tag]
    axB.annotate(
        f"  {tag}  final {ys[-1]:.4f}",
        (xs[-1], ys[-1]),
        textcoords="offset points",
        xytext=(6, dy),
        fontsize=9,
        color=c,
        va="center",
        fontweight="bold",
    )
axB.annotate(
    "base leaves the frame here",
    (20500, 1.212),
    fontsize=8.8,
    color=S2,
    bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=2),
)
axB.text(
    0.30,
    0.62,
    "(c) reaches the same 6e-4 peak as base but leaves it by step 1,802 and stays stable,\n"
    "so a high peak is not on its own fatal. Which schedule is best is a separate question:\n"
    "on best val, (c) sits 0.0059 below B — about 5x the 0.0011 between the repeat pair —\n"
    "but every schedule here is n=1, so treat that as suggestive rather than settled.",
    transform=axB.transAxes,
    fontsize=8.5,
    color=INK2,
    va="top",
    ha="left",
    bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=3),
)

fig.suptitle(
    "The same settings run twice end 0.0758 apart — but their best val differs by 0.0011",
    fontsize=13,
    color=INK,
    x=0.01,
    ha="left",
    y=0.985,
)
fig.text(
    0.01,
    0.947,
    "Same subset (mammal_centered), four schedules plus one repeat of B — dots mark each run's best.\n"
    "The final value does not reproduce; the best-val checkpoint does. That is why the final model is never adopted.",
    fontsize=9.5,
    color=INK2,
    ha="left",
    va="top",
)
fig.tight_layout(rect=[0, 0, 1, 0.925])
fig.savefig("tmp/figs/schedule-comparison.png", dpi=170, facecolor=SURFACE)
print("saved tmp/figs/schedule-comparison.png")
