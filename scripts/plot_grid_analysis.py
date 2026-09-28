"""Draw the grid as relationships rather than as curves.

Three figures, from the one-row-per-arm summary:

  1. the learning rate that holds against model size (log-log, with the band
     between the largest rate that held and the smallest that collapsed),
  2. the loss against processed tokens for each size at its best rate,
  3. where the collapses happened -- the rate against the amount of data seen
     when the run turned.

The curves themselves belong in an appendix; these are what a reader takes away.

    python scripts/plot_grid_analysis.py --summary <file.tsv> --config <file.json>
                                         --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401  (registers a CJK font)
import matplotlib.pyplot as plt

from plot_eval_loss_series import (  # one house style, one place to change it
    INK, INK_2, INK_3, SERIES, TOKENS_LABEL, finish, floors, read_hf, style,
)

MODALITIES = ["protein", "rna", "molnl", "compounds"]
SIZES = ["small", "medium", "large"]
MARKERS = {"small": "o", "medium": "s", "large": "^"}


def load(path: Path):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    for r in rows:
        r["lr"] = float(r["lr"])
        r["params"] = int(r["params"])
        r["collapsed"] = r["collapsed"] == "1"
        r["best_loss"] = float(r["best_loss"])
        r["last_loss"] = float(r["last_loss"])
        r["last_tokens"] = int(r["last_tokens"])
        r["turned_tokens"] = int(r["turned_tokens"]) if r["turned_tokens"] else None
    return rows


def band(rows, modality, size):
    """(held, collapsed, N) -- the largest rate that held, the smallest that did not."""
    arms = [r for r in rows if r["modality"] == modality and r["size"] == size]
    if not arms:
        return None
    held = [r["lr"] for r in arms if not r["collapsed"]]
    gone = [r["lr"] for r in arms if r["collapsed"]]
    return (max(held) if held else None,
            min(gone) if gone else None,
            arms[0]["params"])


def slope(points):
    """Least squares slope of log(lr) on log(N), for points [(N, lr), ...]."""
    if len(points) < 2:
        return None
    xs = [math.log10(n) for n, _ in points]
    ys = [math.log10(v) for _, v in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def fig_lr_vs_size(rows, cfg, out_dir, caption):
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    slopes = {}
    for i, modality in enumerate(MODALITIES):
        pts = []
        for size in SIZES:
            got = band(rows, modality, size)
            if not got:
                continue
            held, gone, n = got
            if held and gone:
                # The boundary is known to lie between the two; draw the interval
                # and take its geometric middle as the point to fit.
                ax.plot([n, n], [held, gone], color=SERIES[i], linewidth=1.4, alpha=.5, zorder=2)
                mid = math.sqrt(held * gone)
                pts.append((n, mid))
                ax.plot([n], [mid], marker=MARKERS[size], markersize=8, color=SERIES[i],
                        markeredgecolor="white", markeredgewidth=1.2, zorder=4)
            elif held:
                # Nothing collapsed: the boundary is above everything tried, so the
                # point is a lower bound and is drawn as an arrow, not as a value.
                ax.annotate("", xy=(n, held * 2.2), xytext=(n, held),
                            arrowprops=dict(arrowstyle="->", color=SERIES[i], lw=1.4))
                ax.plot([n], [held], marker=MARKERS[size], markersize=8,
                        markerfacecolor="white", color=SERIES[i], markeredgewidth=1.6, zorder=4)
        if len(pts) >= 2:
            s = slope(pts)
            slopes[modality] = s
            xs = [p[0] for p in pts]
            lo, hi = min(xs), max(xs)
            ref = pts[0]
            ax.plot([lo, hi], [ref[1] * (lo / ref[0]) ** s, ref[1] * (hi / ref[0]) ** s],
                    color=SERIES[i], linewidth=1.2, linestyle=(0, (5, 3)), zorder=3)
        ax.plot([], [], color=SERIES[i], marker="o", linestyle="-",
                label=f"{modality}" + (f"（傾き {slopes[modality]:+.2f}）" if modality in slopes else ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "パラメータ数", "学習率")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="lower left")
    for size in SIZES:
        ax.plot([], [], marker=MARKERS[size], color=INK_3, linestyle="none", label=size)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=9.5, labelcolor=INK_2,
              loc="lower left", ncol=2)
    fig.tight_layout(rect=(0, .14, 1, .94))
    finish(fig, cfg.get("titles", {}).get("lr_vs_size", ""), caption,
           out_dir / "grid-lr-vs-size.png", bottom=.14)
    return slopes


def fig_loss_vs_tokens(rows, cfg, out_dir, caption):
    """One panel per modality, one line per size, each at the rate that did best."""
    groups = {g["modality"]: g for g in cfg["groups"]}
    mods = [m for m in MODALITIES if m != "compounds"]
    fig, axes = plt.subplots(1, len(mods), figsize=(4.7 * len(mods), 3.6), squeeze=False)
    for i, modality in enumerate(mods):
        ax = axes[0][i]
        for j, size in enumerate(SIZES):
            arms = [r for r in rows if r["modality"] == modality and r["size"] == size
                    and not r["collapsed"]]
            if not arms:
                continue
            pick = min(arms, key=lambda r: r["best_loss"])
            run_dir = Path(groups[modality]["root"]) / pick["run"]
            st, va, _ = read_hf(run_dir)
            if not st:
                continue
            xs = [s * 2_560 * 1_024 for s in st]
            ax.plot(xs, va, color=SERIES[j], linewidth=1.7, zorder=3,
                    label=f"{size}  lr {pick['lr']:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*cfg["xlim"])
        if groups[modality].get("ylim"):
            ax.set_ylim(*groups[modality]["ylim"])
        ax.set_title(f"{modality}", fontsize=10.5, color=INK, loc="left", pad=6)
        style(ax, TOKENS_LABEL, "eval_loss_mask（nats/token）" if i == 0 else "")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: "" if v <= 0 else f"{v / 1e9:g}G" if v < 1e12 else f"{v / 1e12:g}T"))
        if groups[modality].get("floor"):
            floors(ax, [["何も学ばない場合", groups[modality]["floor"]]], "right")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower left")
    fig.tight_layout(rect=(0, .27, 1, .94))
    finish(fig, cfg.get("titles", {}).get("loss_vs_tokens", ""), caption,
           out_dir / "grid-loss-vs-tokens.png", bottom=.27)


def fig_collapse_map(rows, cfg, out_dir, caption):
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    for i, modality in enumerate(MODALITIES):
        for size in SIZES:
            for r in rows:
                if (r["modality"], r["size"]) != (modality, size) or not r["collapsed"]:
                    continue
                if not r["turned_tokens"]:
                    continue
                ax.plot([r["turned_tokens"]], [r["lr"]], marker=MARKERS[size], markersize=9,
                        color=SERIES[i], markeredgecolor="white", markeredgewidth=1.2, zorder=4)
                ax.annotate(f"{r['best_loss']:.2f}", (r["turned_tokens"], r["lr"]),
                            textcoords="offset points", xytext=(0, 9), ha="center",
                            fontsize=7.6, color=SERIES[i])
        ax.plot([], [], color=SERIES[i], marker="o", linestyle="none", label=modality)
    for size in SIZES:
        ax.plot([], [], marker=MARKERS[size], color=INK_3, linestyle="none", label=size)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, TOKENS_LABEL, "学習率")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "" if v <= 0 else f"{v / 1e9:g}G" if v < 1e12 else f"{v / 1e12:g}T"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, .16, 1, .94))
    finish(fig, cfg.get("titles", {}).get("collapse_map", ""), caption,
           out_dir / "grid-collapse-map.png", bottom=.16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load(Path(args.summary))
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    caps = cfg.get("captions", {})
    slopes = fig_lr_vs_size(rows, cfg, out_dir, caps.get("lr_vs_size", ""))
    fig_loss_vs_tokens(rows, cfg, out_dir, caps.get("loss_vs_tokens", ""))
    fig_collapse_map(rows, cfg, out_dir, caps.get("collapse_map", ""))
    for modality, s in slopes.items():
        print(f"  {modality}: 境目 ∝ N^{s:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
