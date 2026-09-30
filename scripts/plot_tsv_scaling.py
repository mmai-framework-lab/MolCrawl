"""Draw scaling figures from a curve TSV another workstream already wrote.

Not every result comes back as a directory of checkpoints. compounds and genome
publish their series as one long TSV, and re-deriving those runs to plot them
would be a second source of truth. This reads the TSV and puts it on the same
axes as everything else: compute in PF-days, log-log, one house style.

Three modes, chosen per figure:

  series  one line per selected series, coloured in order.
  groups  many members drawn thin and their mean drawn thick, one colour per
          group -- for a question asked with 10 seeds rather than one run.
  points  the best value of each series against a chosen x (a learning rate, a
          parameter count), log-log, with the fitted slope.

    python scripts/plot_tsv_scaling.py --config <file.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401  (registers a CJK font)
import matplotlib.pyplot as plt

from plot_eval_loss_series import INK, INK_2, INK_3, SERIES, finish, floors, style, write_tsv

PF_DAY = 8.64e19


def rows_of(path: Path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def select(rows, where: dict):
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def curve(rows, cfg, spec):
    """[(compute in PF-days, value, step)] for one selection, sorted and de-duplicated.

    The step comes along because it is what a log line and a checkpoint name
    carry: a reader checking a point in the figure against a run looks for the
    step, not for the compute.
    """
    step_col = cfg.get("step_column", "step")
    value_col = cfg.get("value_column", "value")
    n = spec.get("params", cfg.get("params"))
    tps = spec.get("tokens_per_step", cfg.get("tokens_per_step"))
    seen = {}
    for r in select(rows, spec["where"]):
        step = int(float(r[step_col]))
        seen[step] = float(r[value_col])
    return [(6 * n * s * tps / PF_DAY, v, s) for s, v in sorted(seen.items()) if s > 0]


def axes_common(ax, cfg, xlabel):
    ax.set_xscale("log")
    ax.set_yscale(cfg.get("yscale", "log"))
    if cfg.get("xlim"):
        ax.set_xlim(*cfg["xlim"])
    if cfg.get("ylim"):
        ax.set_ylim(*cfg["ylim"])
    style(ax, xlabel, cfg["ylabel"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "" if v <= 0 else (f"{v:g}" if v >= 0.01 else f"{v:.3f}")))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "" if v <= 0 else f"{v:g}"))
    if cfg.get("floors"):
        floors(ax, cfg["floors"], cfg.get("floor_side", "right"))


def fig_series(rows, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(cfg.get("width", 11.5), cfg.get("height", 7.0)))
    drawn = []
    for i, spec in enumerate(cfg["series"]):
        pts = curve(rows, cfg, spec)
        if not pts:
            print(f"  ! no rows: {spec['label']}")
            continue
        c = SERIES[i % len(SERIES)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, linewidth=1.9,
                zorder=3, label=spec["label"])
        best = min(p[1] for p in pts)
        ax.annotate(f"{pts[-1][1]:.4f}", (pts[-1][0], pts[-1][1]), textcoords="offset points",
                    xytext=(6, 0), ha="left", va="center", fontsize=9, color=c, fontweight="bold")
        print(f"    {spec['label'][:30]:32s} last={pts[-1][1]:.4f} best={best:.4f} "
              f"C={pts[-1][0]:.3f} PF-days")
        drawn += [{"series": spec["label"], "step": p[2], "pfdays": f"{p[0]:.6f}",
                   "value": f"{p[1]:.6f}"} for p in pts]
    axes_common(ax, cfg, cfg.get("xlabel", "学習に使った計算量（PF-days、C = 6ND）"))
    ax.legend(frameon=False, fontsize=10.5, labelcolor=INK_2, title=cfg.get("legend_title"),
              title_fontsize=9.5, loc=cfg.get("legend_loc", "lower left"))
    fig.tight_layout(rect=(0, cfg.get("rect_bottom", .20), 1, .95))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"], bottom=cfg.get("rect_bottom", .20))
    write_tsv(out_dir / Path(cfg["file"]).with_suffix(".tsv").name, drawn, ["series", "step", "pfdays", "value"])


def fig_groups(rows, cfg, out_dir):
    """Members thin, the group's mean thick: a question answered with many seeds."""
    fig, ax = plt.subplots(figsize=(cfg.get("width", 11.5), cfg.get("height", 7.0)))
    drawn = []
    for i, g in enumerate(cfg["groups"]):
        c = SERIES[i % len(SERIES)]
        members, by_x = [], defaultdict(list)
        for spec in g["members"]:
            pts = curve(rows, cfg, {**spec, "params": g.get("params", cfg.get("params")),
                                    "tokens_per_step": g.get("tokens_per_step",
                                                             cfg.get("tokens_per_step"))})
            if not pts:
                continue
            members.append(pts)
            drawn += [{"group": g["label"], "member": spec["where"].get("subset", ""),
                       "step": p[2], "pfdays": f"{p[0]:.6f}", "value": f"{p[1]:.6f}"}
                      for p in pts]
            for x, v, _step in pts:
                by_x[round(math.log10(x), 3)].append(v)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, linewidth=.7,
                    alpha=.35, zorder=2)
        if not members:
            print(f"  ! no rows: {g['label']}")
            continue
        # The mean is taken over whatever members reached each amount of compute;
        # where they end at different points the line thins out rather than jumps.
        xs = sorted(by_x)
        mean = [(10 ** x, sum(by_x[x]) / len(by_x[x])) for x in xs if len(by_x[x]) >= g.get("min_members", 1)]
        ax.plot([p[0] for p in mean], [p[1] for p in mean], color=c, linewidth=2.2, zorder=4,
                label=f"{g['label']}（{len(members)} 本）")
        drawn += [{"group": g["label"], "member": "（平均）", "step": "",
                   "pfdays": f"{x:.6f}", "value": f"{v:.6f}"} for x, v in mean]
        finals = [m[-1][1] for m in members]
        print(f"    {g['label'][:24]:26s} n={len(members):2} 最終 平均={sum(finals)/len(finals):.4f} "
              f"最小={min(finals):.4f} 最大={max(finals):.4f}")
    axes_common(ax, cfg, cfg.get("xlabel", "学習に使った計算量（PF-days、C = 6ND）"))
    ax.legend(frameon=False, fontsize=10.5, labelcolor=INK_2, title=cfg.get("legend_title"),
              title_fontsize=9.5, loc=cfg.get("legend_loc", "lower left"))
    fig.tight_layout(rect=(0, cfg.get("rect_bottom", .20), 1, .95))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"], bottom=cfg.get("rect_bottom", .20))
    write_tsv(out_dir / Path(cfg["file"]).with_suffix(".tsv").name, drawn,
              ["group", "member", "step", "pfdays", "value"])


def fig_points(rows, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(cfg.get("width", 10.5), cfg.get("height", 6.4)))
    value_col = cfg.get("value_column", "value")
    drawn = []
    for i, ser in enumerate(cfg["series"]):
        c = SERIES[i % len(SERIES)]
        pts = []
        for p in ser["points"]:
            sel = select(rows, p["where"])
            if not sel:
                print(f"  ! no rows: {ser['label']} {p['where']}")
                continue
            lo = min(float(r[value_col]) for r in sel)
            pts.append((p["x"], lo))
            drawn.append({"series": ser["label"], "x": p["x"], "best": f"{lo:.6f}",
                          "n_points": len(sel)})
        pts.sort()
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, linewidth=1.4, alpha=.6, zorder=2)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=ser.get("marker", "o"),
                markersize=9, linestyle="none", color=c, markeredgecolor="white",
                markeredgewidth=1.2, zorder=4, label=ser["label"])
        lo = min(pts, key=lambda p: p[1])
        ax.annotate(f"{lo[1]:.4f}", lo, textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color=c, fontweight="bold")
        print(f"    {ser['label'][:26]:28s} 最良={lo[1]:.4f} @ x={lo[0]:g}")
    ax.set_xscale("log")
    ax.set_yscale(cfg.get("yscale", "log"))
    if cfg.get("xlim"):
        ax.set_xlim(*cfg["xlim"])
    if cfg.get("ylim"):
        ax.set_ylim(*cfg["ylim"])
    style(ax, cfg["xlabel"], cfg["ylabel"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "" if v <= 0 else f"{v:g}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "" if v <= 0 else f"{v:g}"))
    ax.legend(frameon=False, fontsize=10, labelcolor=INK_2, title=cfg.get("legend_title"),
              title_fontsize=9.5, loc=cfg.get("legend_loc", "best"))
    fig.tight_layout(rect=(0, cfg.get("rect_bottom", .18), 1, .95))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"], bottom=cfg.get("rect_bottom", .18))
    write_tsv(out_dir / Path(cfg["file"]).with_suffix(".tsv").name, drawn,
              ["series", "x", "best", "n_points"])


MODE = {"series": fig_series, "groups": fig_groups, "points": fig_points}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cache: dict = {}
    for f in cfg["figures"]:
        src = f.get("tsv", cfg.get("tsv"))
        if src not in cache:
            cache[src] = rows_of(Path(src))
        print(f"{f['file']}:")
        MODE[f["mode"]](cache[src], f, out_dir)
        print(f"  wrote {f['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
