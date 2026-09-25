"""Plot the eval-loss series of a set of runs, from whatever each run recorded.

Two record formats live side by side in this project and they are not the same
measurement, so they are never drawn on one pair of axes:

  nanoGPT  writes "step N: train loss X, val loss Y" to stdout -- both series,
           every eval, in one line.
  HF       writes log_history into checkpoint-*/trainer_state.json. The adoption
           metric is ``eval_loss_mask`` ([MASK] positions only); ``eval_loss`` in
           the same record is the 80/10/10 blend and is not interpretable alone.

Paths live in the config file, not here: this repository is public.

    python scripts/plot_eval_loss_series.py --config <file.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401  (no CJK font is installed system-wide)
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK_2, INK_3 = "#0f161a", "#53626c", "#8695a0"
SURFACE, RULE = "#fcfcfb", "#d7e0e5"

STEP_RE = re.compile(r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)", re.M)


def read_nanogpt(path: Path):
    """(steps, train, val) from a nanoGPT run's stdout."""
    text = Path(path).read_text(errors="ignore")
    rows = [(int(s), float(t), float(v)) for s, t, v in STEP_RE.findall(text)]
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def read_hf(run_dir: Path, key: str = "eval_loss_mask"):
    """(steps, values, meta) from the newest checkpoint's trainer_state.json.

    The newest checkpoint carries the whole log_history, so the series is complete
    even when older checkpoints have been pruned.
    """
    cks = list(Path(run_dir).glob("checkpoint-*/trainer_state.json"))
    if not cks:
        return [], [], {}
    newest = max(cks, key=lambda p: int(p.parent.name.split("-")[1]))
    st = json.loads(newest.read_text())
    pts = [(e["step"], e[key]) for e in st["log_history"] if key in e]
    pts.sort()
    meta = {k: st.get(k) for k in ("max_steps", "num_train_epochs")}
    return [p[0] for p in pts], [p[1] for p in pts], meta


def expand(spec):
    """Resolve a run spec to a list of dicts, deriving labels from the path.

    Listing the runs by pattern keeps the config honest: a run that exists on disk
    cannot be left out of the figure by a typo in a hand-written list.
    """
    import glob as _glob
    out = []
    for pat in spec["glob"] if isinstance(spec.get("glob"), list) else [spec["glob"]]:
        for hit in sorted(_glob.glob(pat)):
            item = {"path": hit}
            for field, rx in spec.get("derive", {}).items():
                m = re.search(rx, hit)
                item[field] = m.group(1) if m else None
            if spec.get("derive_from_text"):
                text = Path(hit).read_text(errors="ignore")
                for field, rx in spec["derive_from_text"].items():
                    m = re.search(rx, text)
                    item[field] = m.group(1) if m else None
            if spec.get("group_map"):
                for rx, name in spec["group_map"]:
                    if re.search(rx, hit):
                        item["group"] = name
                        break
            if any(item.get(k) != v for k, v in spec.get("where", {}).items()):
                continue
            out.append(item)
    return out


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
        ax.spines[s].set_linewidth(.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=.8)
    ax.grid(color=RULE, linewidth=.6, alpha=.7)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def finish(fig, title, caption, out):
    fig.suptitle(title, fontsize=13.5, fontweight="bold", color=INK, x=.012, ha="left", y=.985)
    # Wrap before drawing: bbox_inches="tight" sizes the canvas around a single
    # long line, which stretches the figure to several times its intended width.
    wrapped = "\n".join(textwrap.wrap(caption, width=132, break_long_words=True))
    # Each wrapped line needs its own strip of canvas. A fixed margin fits two and
    # lets a longer caption print over the x axis.
    _lines = wrapped.count("\n") + 1
    if _lines > 2:
        fig.subplots_adjust(bottom=max(.16, .10 + .035 * _lines))
    fig.text(.012, .012, wrapped, fontsize=7.6, color=INK_3, ha="left", va="bottom",
             linespacing=1.6)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def floors(ax, items, side="right"):
    """Model-free reference lines. A loss has no meaning without its floor."""
    lo, hi = ax.get_xlim()
    x, ha, dx = (hi, "right", -3) if side == "right" else (lo, "left", 3)
    for name, v in items:
        ax.axhline(v, color=INK_3, linewidth=1, linestyle=(0, (6, 4)), zorder=1)
        ax.annotate(f"{name} {v}", (x, v), textcoords="offset points", xytext=(dx, 3),
                    ha=ha, fontsize=7.8, color=INK_3)


# --------------------------------------------------------------------------- figures

def best_in_view(steps, values, xlim):
    """Index of the lowest value among the points actually drawn.

    A zoom is a window on the run, and the run's own minimum is usually outside
    it: the compounds 2e-3 arm bottoms at step 1,300 and the last-20% window
    starts at 12,000. Annotating the global minimum there puts the label off the
    axes, where it is silently dropped -- the zoom came out with no marker at all.
    Inside a window, "best" means best in the window.
    """
    idx = range(len(values))
    if xlim:
        lo, hi = xlim
        inside = [i for i in idx if lo <= steps[i] <= hi]
        if inside:
            idx = inside
    return min(idx, key=lambda i: values[i])


def fig_panels_nanogpt(cfg, out_dir):
    """Small multiples: one panel per size, learning rates coloured inside it."""
    groups = cfg["panels"]
    n = len(groups)
    cols = cfg.get("cols") or min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cfg.get("panel_w", 5.7) * cols,
                                      cfg.get("panel_h", 3.0) * rows + .8),
                             squeeze=False)
    lrs = cfg["lr_order"]
    xmax = 0
    for i, g in enumerate(groups):
        ax = axes[i // cols][i % cols]
        seen = set()
        # A run that was requeued writes a second log that picks up from the
        # checkpoint, so one arm can span several files. Merge them by step before
        # drawing, or the same arm appears twice.
        arms = {}
        for run in expand(g):
            if run.get("lr") not in lrs:
                continue
            st, tr, va = read_nanogpt(Path(run["path"]))
            if not st:
                continue
            key = run.get("arm") or run["path"]
            arms.setdefault(key, {}).update(dict(zip(st, zip(tr, va))))
            arms[key + "\0lr"] = run["lr"]
        merged = []
        for key in [k for k in arms if not k.endswith("\0lr")]:
            pts = sorted(arms[key].items())
            merged.append({"lr": arms[key + "\0lr"],
                           "st": [p[0] for p in pts],
                           "tr": [p[1][0] for p in pts],
                           "va": [p[1][1] for p in pts]})
        for run in sorted(merged, key=lambda r: lrs.index(r["lr"])):
            st, tr, va = run["st"], run["tr"], run["va"]
            if st[-1] < g.get("min_last_step", 0):
                continue
            xmax = max(xmax, st[-1])
            c = SERIES[lrs.index(run["lr"])]
            if cfg.get("show_train"):
                ax.plot(st, tr, color=c, linewidth=1, linestyle=(0, (4, 3)), alpha=.5, zorder=2)
            ax.plot(st, va, color=c, linewidth=1.6, alpha=.95, zorder=3,
                    label=run["lr"] if run["lr"] not in seen else None)
            seen.add(run["lr"])
            # Where a run bottoms out is the reading of interest once a schedule
            # runs past its minimum: the 30-epoch compounds sweep turns back up,
            # and the turn is at a different iteration for every learning rate.
            best_i = best_in_view(st, va, g.get("xlim"))
            if cfg.get("mark_best"):
                ax.plot([st[best_i]], [va[best_i]], marker="o", markersize=4.5, color=c,
                        markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=5)
            if cfg.get("annotate_step"):
                dx, dy = g.get("min_offsets", {}).get(run["lr"], [0, -12])
                ax.annotate(f"{va[best_i]:.4f} @ {st[best_i]:,}", (st[best_i], va[best_i]),
                            textcoords="offset points", xytext=(dx, dy), ha="center",
                            fontsize=7.4, color=c, fontweight="bold", zorder=6)
            print(f"    {g['title'][:18]:20s} lr={run['lr']:8s} last={st[-1]:6d} "
                  f"best_val={va[best_i]:.4f}@{st[best_i]}")
        ax.set_title(g["title"], fontsize=cfg.get("title_size", 10.5), color=INK,
                     loc="left", pad=6)
        if cfg.get("xticks"):
            ax.set_xticks(cfg["xticks"])
            ax.set_xticklabels(cfg.get("xticklabels") or [str(t) for t in cfg["xticks"]])
        if g.get("ylim"):
            ax.set_ylim(*g["ylim"])
        if g.get("xlim"):
            ax.set_xlim(*g["xlim"])
        style(ax, cfg["xlabel"] if i // cols == rows - 1 else "",
              cfg["ylabel"] if i % cols == 0 else "")
        if cfg.get("floors"):
            floors(ax, cfg["floors"], cfg.get("floor_side", "right"))
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda k: lrs.index(labels[k]))
        ax.legend([handles[k] for k in order], [labels[k] for k in order],
                  frameon=False, fontsize=8.5, labelcolor=INK_2, title=cfg.get("legend_title"),
                  title_fontsize=8, loc=cfg.get("legend_loc", "upper right"))
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout(rect=(0, cfg.get("rect_bottom", .05), 1, .955))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"])


def fig_many_hf(cfg, out_dir):
    """Many runs, few colours: colour carries the group, never the individual run."""
    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    seen, xmax = set(), 0
    runs = expand(cfg)
    print(f"  {len(runs)} runs matched")
    for run in runs:
        st, va, _ = read_hf(Path(run["path"]), cfg.get("key", "eval_loss_mask"))
        if not st:
            print(f"  ! no series: {run['path']}")
            continue
        xmax = max(xmax, st[-1])
        gi = cfg["group_order"].index(run.get("group") or cfg["group_order"][0])
        g = run.get("group") or cfg["group_order"][0]
        ax.plot(st, va, color=SERIES[gi], linewidth=1.1,
                alpha=cfg.get("alpha", .75), zorder=3,
                label=g if g not in seen else None)
        seen.add(g)
        print(f"    {Path(run['path']).name[:46]:48s} last={st[-1]:7d} best={min(va):.4f}@{st[va.index(min(va))]}")
    if cfg.get("ylim"):
        ax.set_ylim(*cfg["ylim"])
    if cfg.get("xlim"):
        ax.set_xlim(*cfg["xlim"])
    style(ax, cfg["xlabel"], cfg["ylabel"])
    if cfg.get("floors"):
        floors(ax, cfg["floors"], cfg.get("floor_side", "right"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc=cfg.get("legend_loc", "upper right"))
    fig.tight_layout(rect=(0, .06, 1, .94))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"])


def write_tsv(path, rows, columns):
    """Every evaluation point, so a figure can be checked against the numbers.

    A curve at this scale cannot be read to the fourth decimal, which is where the
    difference between two arms of a grid can live.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")
    print(f"  wrote {Path(path).name} ({len(rows)} rows)")


def fig_lines_hf(cfg, out_dir):
    """A handful of runs, one colour each."""
    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    tsv_rows = []
    for i, run in enumerate(cfg["runs"]):
        st, va, _ = read_hf(Path(run["dir"]), cfg.get("key", "eval_loss_mask"))
        if not st:
            print(f"  ! no series: {run['dir']}")
            continue
        ax.plot(st, va, color=SERIES[i], linewidth=1.8, zorder=3, label=run["label"])
        best_i = best_in_view(st, va, cfg.get("xlim"))
        print(f"    {run['label'][:30]:32s} last={st[-1]:7d} best={va[best_i]:.4f}@{st[best_i]}")
        dx, dy = cfg.get("min_offsets", [[0, -13]] * len(cfg["runs"]))[i]
        # The step belongs beside the value: a minimum in the middle of a run is a
        # different statement from one at the last evaluation, and the two arms of
        # this grid differ in which they are.
        text = (f"{va[best_i]:.4f} @ {st[best_i]:,}" if cfg.get("annotate_step")
                else f"{va[best_i]:.4f}")
        if cfg.get("mark_best"):
            ax.plot([st[best_i]], [va[best_i]], marker="o", markersize=5,
                    color=SERIES[i], markeredgecolor=SURFACE, markeredgewidth=1.2,
                    zorder=5)
        ax.annotate(text, (st[best_i], va[best_i]),
                    textcoords="offset points", xytext=(dx, dy), ha="center",
                    fontsize=8, color=SERIES[i], fontweight="bold")
        for step, value in zip(st, va):
            row = {k: v for k, v in run.items() if k != "dir"}
            row.update({"step": step, cfg.get("key", "eval_loss_mask"): f"{value:.6f}"})
            tsv_rows.append(row)
    # A run that falls by two orders of magnitude spends most of a linear axis
    # flat against the bottom, where the part worth reading is.
    if cfg.get("yscale"):
        ax.set_yscale(cfg["yscale"])
    if cfg.get("ylim"):
        ax.set_ylim(*cfg["ylim"])
    if cfg.get("xlim"):
        ax.set_xlim(*cfg["xlim"])
    style(ax, cfg["xlabel"], cfg["ylabel"])
    if cfg.get("floors"):
        floors(ax, cfg["floors"], cfg.get("floor_side", "right"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc=cfg.get("legend_loc", "upper right"))
    fig.tight_layout(rect=(0, .06, 1, .94))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"])
    if cfg.get("tsv"):
        write_tsv(out_dir / cfg["tsv"], tsv_rows,
                  cfg.get("tsv_columns",
                          ["size", "lr", "step", cfg.get("key", "eval_loss_mask"), "segment"]))


KIND = {"panels_nanogpt": fig_panels_nanogpt, "many_hf": fig_many_hf, "lines_hf": fig_lines_hf}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for f in cfg["figures"]:
        print(f"{f['file']}:")
        KIND[f["kind"]](f, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
