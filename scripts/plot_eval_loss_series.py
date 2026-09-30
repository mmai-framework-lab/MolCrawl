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
import glob
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401  (no CJK font is installed system-wide)
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
# One wording for the axis every figure in this project now shares.
TOKENS_LABEL = "処理トークン数"
PF_DAY = 8.64e19  # 1 petaflop/s-day = 1e15 operations/s x 86,400 s
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


def tokens_axis(ax, cfg):
    """Put the x axis in processed tokens, on the range every panel shares.

    A step is not the same amount of work at two model sizes -- large does about
    three times the compute of small in one step -- so panels whose x axis is the
    step number compare different points. Tokens (step x tokens/step) are the same
    quantity everywhere, and one range across the panels makes the comparison the
    reader is invited to make an honest one.
    """
    if not cfg.get("x_tokens_per_step"):
        return
    ax.set_xscale(cfg.get("xscale", "log"))
    if cfg.get("xlim"):
        ax.set_xlim(*cfg["xlim"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "" if v <= 0 else
        f"{v / 1e12:g}T" if v >= 1e12 else
        f"{v / 1e9:g}G" if v >= 1e9 else
        f"{v / 1e6:g}M"))


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


def finish(fig, title, caption, out, bottom=0.0):
    fig.suptitle(title, fontsize=13.5, fontweight="bold", color=INK, x=.012, ha="left", y=.985)
    # Wrap before drawing: bbox_inches="tight" sizes the canvas around a single
    # long line, which stretches the figure to several times its intended width.
    wrapped = "\n".join(textwrap.wrap(caption, width=132, break_long_words=True))
    # Each wrapped line needs its own strip of canvas. A fixed margin fits two and
    # lets a longer caption print over the x axis.
    _lines = wrapped.count("\n") + 1
    if _lines > 2:
        # Never below what the caller asked for: this runs after tight_layout and
        # would otherwise pull the axes back down onto a long caption.
        fig.subplots_adjust(bottom=max(.16, bottom, .10 + .035 * _lines))
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

def x_scaler(cfg, params=None):
    """Turn a step index into the x quantity a figure asks for.

    Three axes answer three questions and the choice is not cosmetic: a step is
    not the same work at two sizes. Steps stay available for a single run read on
    its own; tokens compare data efficiency; PF-days (C = 6ND over 8.64e19)
    compare compute, and are what a claim crossing sizes has to be read on.

    6ND omits the attention term, which at sequence length 1,024 is 13-18 % of the
    true cost. It cancels in a ratio between sizes; an absolute figure carries the
    caveat in its caption.
    """
    per_step = cfg.get("x_tokens_per_step") or 1
    if cfg.get("x_pfdays"):
        n = params if params is not None else cfg.get("params")
        if not n:
            raise SystemExit("x_pfdays needs params (non-embedding) on the figure or the run")
        return lambda step: 6 * n * step * per_step / PF_DAY
    return lambda step: step * per_step


def best_overall(steps, values, xlim):
    """Index of the run's lowest value, and whether it falls inside the drawn window.

    The mark is the run's own best, not the best of whatever slice is on screen: a
    zoom is a window on one run, and two figures of the same run marking different
    points invites the reader to take them as different runs. Where the best is
    outside the window there is nothing to mark, so the caller writes it into the
    caption instead -- hence the second return value.
    """
    best = min(range(len(values)), key=lambda i: values[i])
    inside = True
    if xlim:
        lo, hi = xlim
        inside = lo <= steps[best] <= hi
    return best, inside


def offscreen_note(label, step, value):
    """The footnote a figure carries when a run's best is off the drawn window."""
    return f"{label} の最良はこの窓より前、{value:.4f} @ step {step:,}。"


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
    offscreen = []
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
            raw_steps = st
            if raw_steps[-1] < g.get("min_last_step", 0):
                continue
            to_x = x_scaler(cfg, g.get("params"))
            st = [to_x(x) for x in st]
            xmax = max(xmax, st[-1])
            c = SERIES[lrs.index(run["lr"])]
            if cfg.get("show_train"):
                ax.plot(st, tr, color=c, linewidth=1, linestyle=(0, (4, 3)), alpha=.5, zorder=2)
            ax.plot(st, va, color=c, linewidth=1.6, alpha=.95, zorder=3,
                    label=(cfg.get("lr_labels", {}).get(run["lr"], run["lr"])
                           if run["lr"] not in seen else None))
            seen.add(run["lr"])
            # Where a run bottoms out is the reading of interest once a schedule
            # runs past its minimum: the 30-epoch compounds sweep turns back up,
            # and the turn is at a different iteration for every learning rate.
            best_i, best_shown = best_overall(st, va, g.get("xlim"))
            if not best_shown:
                offscreen.append(offscreen_note(f"{g['title'].split('（')[0]} lr {run['lr']}",
                                                st[best_i], va[best_i]))
            if cfg.get("mark_best") and best_shown:
                ax.plot([st[best_i]], [va[best_i]], marker="o", markersize=4.5, color=c,
                        markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=5)
            if cfg.get("annotate_step") and best_shown:
                dx, dy = g.get("min_offsets", {}).get(run["lr"], [0, -12])
                _at = (f"{st[best_i]:.3f} PF-days" if cfg.get("x_pfdays")
                       else f"{raw_steps[best_i]:,}")
                ax.annotate(f"{va[best_i]:.4f} @ {_at}", (st[best_i], va[best_i]),
                            textcoords="offset points", xytext=(dx, dy), ha="center",
                            fontsize=7.4, color=c, fontweight="bold", zorder=6)
            print(f"    {g['title'][:18]:20s} lr={run['lr']:8s} last={raw_steps[-1]:6d} "
                  f"best_val={va[best_i]:.4f}@step {raw_steps[best_i]}")
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
        tokens_axis(ax, cfg)
        if cfg.get("floors"):
            floors(ax, cfg["floors"], cfg.get("floor_side", "right"))
        handles, labels = ax.get_legend_handles_labels()
        raw = {cfg.get("lr_labels", {}).get(x, x): x for x in lrs}
        order = sorted(range(len(labels)), key=lambda k: lrs.index(raw[labels[k]]))
        # Where a panel has room differs between panels: a curve that rises at the
        # right leaves the top-right corner occupied.
        ax.legend([handles[k] for k in order], [labels[k] for k in order],
                  frameon=False, fontsize=8.5, labelcolor=INK_2, title=cfg.get("legend_title"),
                  title_fontsize=8, loc=g.get("legend_loc") or cfg.get("legend_loc", "upper right"))
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout(rect=(0, cfg.get("rect_bottom", .05), 1, .955))
    caption = cfg["caption"]
    if offscreen:
        caption += "  " + " ".join(offscreen)
    finish(fig, cfg["title"], caption, out_dir / cfg["file"],
           bottom=cfg.get("rect_bottom", 0))


def fig_panels_hf(cfg, out_dir):
    """Small multiples over HF runs: one panel per size, learning rates coloured.

    The nanoGPT twin of this reads one stdout file per run; here each run is a
    directory and the series comes from the newest checkpoint's trainer_state.json,
    so a run that spans several segments still yields one curve.
    """
    groups = cfg["panels"]
    n = len(groups)
    cols = cfg.get("cols") or min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cfg.get("panel_w", 5.7) * cols,
                                      cfg.get("panel_h", 3.0) * rows + .8),
                             squeeze=False)
    key = cfg.get("key", "eval_loss_mask")
    tsv_rows = []
    for i, g in enumerate(groups):
        ax = axes[i // cols][i % cols]
        # A panel may carry its own rates: when a grid is filled in, the rates that
        # bracket the collapse differ per size, and there are more of them in total
        # than the palette has slots.
        lrs = g.get("lr_order") or cfg["lr_order"]
        runs = [r for r in expand(g) if r.get("lr") in lrs]
        for run in sorted(runs, key=lambda r: lrs.index(r["lr"])):
            st, va, _ = read_hf(Path(run["path"]), key)
            if not st:
                print(f"  ! no series: {run['path']}")
                continue
            c = SERIES[lrs.index(run["lr"])]
            scale = cfg.get("x_tokens_per_step") or 1
            steps, st = st, [x * scale for x in st]
            # A path spells a rate as "lr1e4"; the legend has to read "1e-4".
            ax.plot(st, va, color=c, linewidth=1.6, alpha=.95, zorder=3,
                    label=cfg.get("lr_labels", {}).get(run["lr"], run["lr"]))
            print(f"    {g['title'][:18]:20s} lr={run['lr']:8s} last={st[-1]:7d} "
                  f"last_val={va[-1]:.4f} best={min(va):.4f}@{st[va.index(min(va))]}")
            # The TSV keeps the step, which is what a log line and a checkpoint
            # name carry, and adds the tokens the figure is drawn against.
            for step, value in zip(steps, va):
                tsv_rows.append({"panel": g["title"], "lr": run["lr"], "step": step,
                                 "tokens": step * scale, key: f"{value:.6f}"})
        ax.set_title(g["title"], fontsize=cfg.get("title_size", 10.5), color=INK,
                     loc="left", pad=6)
        if cfg.get("yscale"):
            ax.set_yscale(cfg["yscale"])
        if g.get("ylim"):
            ax.set_ylim(*g["ylim"])
        if g.get("xlim"):
            ax.set_xlim(*g["xlim"])
        style(ax, cfg["xlabel"] if i // cols == rows - 1 else "",
              cfg["ylabel"] if i % cols == 0 else "")
        tokens_axis(ax, cfg)
        # The floor belongs in every panel: a curve resting on it is the whole
        # reading of a collapsed arm, and it differs per modality.
        if g.get("floors") or cfg.get("floors"):
            floors(ax, g.get("floors") or cfg["floors"], cfg.get("floor_side", "right"))
        handles, labels = ax.get_legend_handles_labels()
        raw = {cfg.get("lr_labels", {}).get(x, x): x for x in lrs}
        order = sorted(range(len(labels)), key=lambda k: lrs.index(raw[labels[k]]))
        # Where a panel has room differs between panels: a curve that rises at the
        # right leaves the top-right corner occupied.
        ax.legend([handles[k] for k in order], [labels[k] for k in order],
                  frameon=False, fontsize=8.5, labelcolor=INK_2, title=cfg.get("legend_title"),
                  title_fontsize=8, loc=g.get("legend_loc") or cfg.get("legend_loc", "upper right"))
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout(rect=(0, cfg.get("rect_bottom", .05), 1, .955))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"],
           bottom=cfg.get("rect_bottom", 0))
    if cfg.get("tsv"):
        write_tsv(out_dir / cfg["tsv"], tsv_rows, ["panel", "lr", "step", "tokens", key])


def fig_scaling(cfg, out_dir):
    """One large axes, one line per model size, x in PF-days.

    The figure a scaling claim is read from: sizes on top of each other against
    the compute each of them used, so the reader compares equal work rather than
    equal steps. Colour carries the size and nothing else, and the legend carries
    the parameter count that the compute was computed from.
    """
    fig, ax = plt.subplots(figsize=(cfg.get("width", 11.0), cfg.get("height", 6.6)))
    tsv_rows, curves = [], []
    for i, run in enumerate(cfg["sizes"]):
        if run.get("kind", "nanogpt") == "nanogpt":
            merged = {}
            for path in sorted(sum([glob.glob(p) for p in run["glob"]], [])):
                st, _tr, va = read_nanogpt(Path(path))
                # A run that was requeued writes a second log picking up from the
                # checkpoint; merge on the step or the arm is drawn twice.
                merged.update(dict(zip(st, va)))
            pts = sorted(merged.items())
            steps = [p[0] for p in pts]
            values = [p[1] for p in pts]
        else:
            steps, values, _ = read_hf(Path(run["dir"]), cfg.get("key", "eval_loss_mask"))
        if not steps:
            print(f"  ! no series: {run['label']}")
            continue
        tokens_per_step = cfg.get("tokens_per_step", 2_621_440)
        xs = [6 * run["params"] * s * tokens_per_step / PF_DAY for s in steps]
        keep = [(x, v) for x, v in zip(xs, values) if x > 0 and v > 0]
        c = SERIES[i % len(SERIES)]
        curves.append(keep)
        ax.plot([k[0] for k in keep], [k[1] for k in keep], color=c, linewidth=1.9,
                zorder=3, label=run["label"])
        best = min(values)
        ax.annotate(f"{values[-1]:.4f}", (keep[-1][0], keep[-1][1]),
                    textcoords="offset points", xytext=(6, 0), ha="left", va="center",
                    fontsize=9, color=c, fontweight="bold")
        print(f"    {run['label'][:26]:28s} last={values[-1]:.4f} best={best:.4f} "
              f"C={keep[-1][0]:.2f} PF-days")
        for step, value in zip(steps, values):
            tsv_rows.append({"size": run["label"], "step": step,
                             "tokens": step * tokens_per_step,
                             "pfdays": f"{6 * run['params'] * step * tokens_per_step / PF_DAY:.4f}",
                             "loss": f"{value:.6f}"})
    if cfg.get("envelope") and curves:
        # The compute-optimal frontier: at each amount of compute, the best any
        # size reached by then. This lower hull is what a scaling law is fitted
        # to -- an individual size's curve leaves it once a larger one overtakes.
        allpts = sorted(p for c in curves for p in c)
        hull_x, hull_y, run_min = [], [], float("inf")
        for x, v in allpts:
            if v < run_min:
                run_min = v
                hull_x.append(x)
                hull_y.append(v)
        ax.plot(hull_x, hull_y, color=INK_3, linewidth=1.3, linestyle=(0, (6, 4)),
                zorder=2, label="到達可能な最小（各計算量での最良）")
        print(f"    包絡線: {hull_y[0]:.4f} @ {hull_x[0]:.3f} PF-days "
              f"→ {hull_y[-1]:.4f} @ {hull_x[-1]:.2f} PF-days")
    ax.set_xscale("log")
    ax.set_yscale(cfg.get("yscale", "log"))
    if cfg.get("xlim"):
        ax.set_xlim(*cfg["xlim"])
    if cfg.get("ylim"):
        ax.set_ylim(*cfg["ylim"])
    style(ax, cfg.get("xlabel", "学習に使った計算量（PF-days、C = 6ND）"), cfg["ylabel"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "" if v <= 0 else (f"{v:g}" if v >= 0.01 else f"{v:.3f}")))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "" if v <= 0 else f"{v:g}"))
    if cfg.get("floors"):
        floors(ax, cfg["floors"], cfg.get("floor_side", "right"))
    ax.legend(frameon=False, fontsize=10.5, labelcolor=INK_2, title=cfg.get("legend_title"),
              title_fontsize=9.5, loc=cfg.get("legend_loc", "lower left"))
    fig.tight_layout(rect=(0, cfg.get("rect_bottom", .16), 1, .95))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"],
           bottom=cfg.get("rect_bottom", .16))
    if cfg.get("tsv"):
        write_tsv(out_dir / cfg["tsv"], tsv_rows, ["size", "step", "tokens", "pfdays", "loss"])


def fig_points(cfg, out_dir):
    """Best loss against model size, one line per modality, with the fitted slope.

    The second figure a scaling claim needs: the curves reduced to their end
    points, so the reader sees whether loss follows a power law in N. A run that
    has not finished is drawn hollow with a downward arrow -- its point can still
    move, and a filled marker would invite a comparison it cannot support.
    """
    import math
    fig, ax = plt.subplots(figsize=(cfg.get("width", 10.5), cfg.get("height", 6.4)))
    for i, ser in enumerate(cfg["series"]):
        # Absolute losses from different corpora share no scale, so a common axis
        # flattens every trend. Dividing by the smallest model's loss puts the
        # shape of each series where it can be read against the others.
        base = ser["points"][0][1] if cfg.get("relative") else 1.0
        done = [(n, v / base) for n, v, fin in ser["points"] if fin]
        running = [(n, v / base) for n, v, fin in ser["points"] if not fin]
        c = SERIES[i % len(SERIES)]
        xs = [p[0] for p in ser["points"]]
        ys = [p[1] / base for p in ser["points"]]
        ax.plot(xs, ys, color=c, linewidth=1.3, alpha=.55, zorder=2)
        if done:
            ax.plot([p[0] for p in done], [p[1] for p in done], marker=ser.get("marker", "o"),
                    markersize=9, linestyle="none", color=c, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=4)
        for n, v in running:
            ax.plot([n], [v], marker=ser.get("marker", "o"), markersize=9, linestyle="none",
                    markerfacecolor="white", color=c, markeredgewidth=1.8, zorder=4)
            ax.annotate("", xy=(n, v * 0.985), xytext=(n, v),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.3))
        label = ser["label"]
        if len(done) >= 2:
            lx = [math.log10(p[0]) for p in done]
            ly = [math.log10(p[1]) for p in done]
            mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
            den = sum((x - mx) ** 2 for x in lx)
            slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / den if den else 0
            label += f"（傾き {slope:+.3f}）"
        ax.plot([], [], color=c, marker=ser.get("marker", "o"), linestyle="-", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if cfg.get("xlim"):
        ax.set_xlim(*cfg["xlim"])
    if cfg.get("ylim"):
        ax.set_ylim(*cfg["ylim"])
    style(ax, cfg.get("xlabel", "非埋め込みパラメータ数"), cfg["ylabel"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "" if v <= 0 else f"{v / 1e6:g}M" if v < 1e9 else f"{v / 1e9:g}B"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "" if v <= 0 else f"{v:g}"))
    ax.legend(frameon=False, fontsize=10, labelcolor=INK_2, title=cfg.get("legend_title"),
              title_fontsize=9.5, loc=cfg.get("legend_loc", "best"))
    fig.tight_layout(rect=(0, cfg.get("rect_bottom", .18), 1, .95))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"],
           bottom=cfg.get("rect_bottom", .18))


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
    offscreen = []
    for i, run in enumerate(cfg["runs"]):
        st, va, _ = read_hf(Path(run["dir"]), cfg.get("key", "eval_loss_mask"))
        if not st:
            print(f"  ! no series: {run['dir']}")
            continue
        to_x = x_scaler(cfg, run.get("params"))
        steps, st = st, [to_x(x) for x in st]
        ax.plot(st, va, color=SERIES[i], linewidth=1.8, zorder=3, label=run["label"])
        best_i, best_shown = best_overall(st, va, cfg.get("xlim"))
        if not best_shown:
            offscreen.append(offscreen_note(run["label"], st[best_i], va[best_i]))
        print(f"    {run['label'][:30]:32s} last_step={steps[-1]:7d} "
              f"best={va[best_i]:.4f}@step {steps[best_i]}")
        dx, dy = cfg.get("min_offsets", [[0, -13]] * len(cfg["runs"]))[i]
        # The step belongs beside the value: a minimum in the middle of a run is a
        # different statement from one at the last evaluation, and the two arms of
        # this grid differ in which they are.
        _at = (f"{st[best_i]:.3f} PF-days" if cfg.get("x_pfdays")
               else f"step {steps[best_i]:,}")
        text = (f"{va[best_i]:.4f} @ {_at}" if cfg.get("annotate_step")
                else f"{va[best_i]:.4f}")
        if cfg.get("mark_best") and best_shown:
            ax.plot([st[best_i]], [va[best_i]], marker="o", markersize=5,
                    color=SERIES[i], markeredgecolor=SURFACE, markeredgewidth=1.2,
                    zorder=5)
        if best_shown:
            ax.annotate(text, (st[best_i], va[best_i]),
                        textcoords="offset points", xytext=(dx, dy), ha="center",
                        fontsize=8, color=SERIES[i], fontweight="bold")
        for step, value in zip(steps, va):
            row = {k: v for k, v in run.items() if k != "dir"}
            row.update({"step": step,
                        "tokens": step * (cfg.get("x_tokens_per_step") or 1),
                        cfg.get("key", "eval_loss_mask"): f"{value:.6f}"})
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
    tokens_axis(ax, cfg)
    if cfg.get("floors"):
        floors(ax, cfg["floors"], cfg.get("floor_side", "right"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc=cfg.get("legend_loc", "upper right"))
    fig.tight_layout(rect=(0, .06, 1, .94))
    caption = cfg["caption"]
    if offscreen:
        caption += "  " + " ".join(offscreen)
    finish(fig, cfg["title"], caption, out_dir / cfg["file"])
    if cfg.get("tsv"):
        write_tsv(out_dir / cfg["tsv"], tsv_rows,
                  cfg.get("tsv_columns",
                          ["size", "lr", "step", cfg.get("key", "eval_loss_mask"), "segment"]))


KIND = {"panels_nanogpt": fig_panels_nanogpt, "panels_hf": fig_panels_hf,
        "scaling": fig_scaling, "points": fig_points,
        "many_hf": fig_many_hf, "lines_hf": fig_lines_hf}


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
