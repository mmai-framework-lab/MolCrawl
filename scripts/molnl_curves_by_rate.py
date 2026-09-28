#!/usr/bin/env python3
"""nanoGPT curves in panels, one panel per learning rate, with a logarithmic y axis.

plot_eval_loss_series.py draws this shape for HF runs and for nanoGPT runs, and its HF
builder honours a "yscale" in the config while its nanoGPT builder does not. A
molecule_nat_lang GPT-2 run starts at 10.96 and ends near 0.58, so on a linear axis every
curve lies along the bottom of the panel and the comparison the figure exists for is
invisible. This draws the nanoGPT case with the axis the data needs, borrowing the house
style from that module so the two look like one set of figures.

It reads the same config; only the "figures" entries of kind "panels_nanogpt" are drawn.

    python scripts/molnl_curves_by_rate.py --config <file.json> --out-dir <dir>
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
import japanize_matplotlib  # noqa: F401,E402  (registers a CJK font)
import matplotlib.pyplot as plt  # noqa: E402

def house_style(from_dir=None):
    """Load plot_eval_loss_series, from a named directory when one is given.

    Python puts the running script's own directory first on sys.path, so an older copy
    of the module sitting beside this file wins over anything PYTHONPATH says. The
    tokens axis lives on a branch that is not merged yet, and importing the wrong copy
    fails with a missing name rather than a wrong figure -- but only because the name is
    missing. --style-from makes the choice explicit.
    """
    if from_dir:
        import importlib.util
        path = os.path.join(from_dir, "plot_eval_loss_series.py")
        spec = importlib.util.spec_from_file_location("plot_eval_loss_series_house", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    import plot_eval_loss_series as module
    return module


def draw(cfg, out_dir, house):
    INK, INK_2, SERIES = house.INK, house.INK_2, house.SERIES
    finish, floors, read_nanogpt = house.finish, house.floors, house.read_nanogpt
    style, tokens_axis = house.style, house.tokens_axis

    panels = cfg["panels"]
    cols = cfg.get("cols") or min(len(panels), 2)
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(cfg.get("panel_w", 4.3) * cols,
                                      cfg.get("panel_h", 3.3) * rows + .8))
    order = cfg["lr_order"]
    scale = cfg.get("x_tokens_per_step") or 1
    for i, panel in enumerate(panels):
        ax = axes[i // cols][i % cols]
        series = {}
        for pattern in (panel["glob"] if isinstance(panel["glob"], list) else [panel["glob"]]):
            for path in sorted(glob.glob(pattern)):
                # The config names the field as plot_eval_loss_series does; here only
                # the captured value is used, as the series key.
                name = None
                for rx in panel.get("derive", {}).values():
                    m = re.search(rx, path)
                    if m:
                        name = m.group(1)
                if name is None or name not in order:
                    continue
                st, _train, val = read_nanogpt(Path(path))
                if st:
                    # A requeued run writes a second log continuing from a checkpoint.
                    series.setdefault(name, {}).update(dict(zip(st, val)))
        for name in order:
            if name not in series:
                continue
            pts = sorted(series[name].items())
            xs = [s * scale for s, _ in pts]
            ys = [v for _, v in pts]
            ax.plot(xs, ys, color=SERIES[order.index(name)], linewidth=1.6, zorder=3,
                    label=cfg.get("lr_labels", {}).get(name, name))
            print(f"    {panel['title'][:22]:24s} {name:<7} last={ys[-1]:.4f} "
                  f"best={min(ys):.4f}@{xs[ys.index(min(ys))]:,.0f}")
        ax.set_yscale(cfg.get("yscale", "log"))
        if panel.get("ylim"):
            ax.set_ylim(*panel["ylim"])
        ax.set_title(panel["title"], fontsize=cfg.get("title_size", 10.5), color=INK,
                     loc="left", pad=6)
        style(ax, cfg["xlabel"] if i // cols == rows - 1 else "",
              cfg["ylabel"] if i % cols == 0 else "")
        tokens_axis(ax, cfg)
        if panel.get("floors") or cfg.get("floors"):
            floors(ax, panel.get("floors") or cfg["floors"], cfg.get("floor_side", "right"))
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, title=cfg.get("legend_title"),
                  title_fontsize=8, loc=panel.get("legend_loc", "lower left"))
    for j in range(len(panels), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout(rect=(0, cfg.get("rect_bottom", .05), 1, .955))
    finish(fig, cfg["title"], cfg["caption"], out_dir / cfg["file"],
           bottom=cfg.get("rect_bottom", 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--style-from", help="directory holding plot_eval_loss_series.py")
    a = ap.parse_args()
    house = house_style(a.style_from)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    drawn = 0
    for figure in cfg["figures"]:
        if figure.get("kind") != "panels_nanogpt":
            continue
        print(f"{figure['file']}:")
        draw(figure, out_dir, house)
        drawn += 1
    if not drawn:
        print("no panels_nanogpt figures in this config")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
