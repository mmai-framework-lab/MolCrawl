"""Render a progress deck (.pptx) from a JSON content file.

The numbers live in the JSON, not here: this repository is public, so the script
carries layout only and is pointed at a content file kept outside any checkout.

    python scripts/build_review12_deck.py --content <file.json> --out-dir <dir>

The JSON holds one or more decks, each a list of slides. Every slide has a
``kind`` that picks a renderer; all of them take ``title`` (written as an
assertion, not a label), optional ``bullets``, an optional ``table``, an optional
``chart`` spec, and an optional ``note`` for the conditions a loss figure must
carry (step / epoch / dataset / global batch / tokens seen).

Charts are rendered to PNG with matplotlib and placed as pictures, so the deck
opens without any external dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401  (registers a CJK font; no CJK font is installed system-wide)
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

# Categorical slots 1-3 of the reference palette, in fixed order. Never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
SURFACE = "#fcfcfb"
RULE = "#d8d7d2"

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.62)
BODY_W = SLIDE_W - 2 * MARGIN


def _rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h.lstrip("#").upper())


def _box(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def _run(para, text, size, *, bold=False, color=INK, space_after=0):
    para.space_after = Pt(space_after)
    r = para.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = _rgb(color)
    return r


# --------------------------------------------------------------------------- charts


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _finish(fig, path: Path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _chart_dumbbell(spec, path):
    """Two measurements per category, joined by a rule.

    A bar chart cannot be used here: the interesting difference is ~0.02 on a
    value near 2.5, so the scale must be truncated -- and a truncated baseline
    makes bar *area* lie. Dots have no area, so the same scale stays honest.
    """
    cats, series = spec["categories"], spec["series"]
    fig, ax = plt.subplots(figsize=(9.2, 2.6))
    ys = list(range(len(cats)))
    a, b = series[0]["values"], series[1]["values"]
    for y, va, vb in zip(ys, a, b):
        ax.plot([va, vb], [y, y], color=RULE, linewidth=2.4, zorder=2,
                solid_capstyle="round")
        ax.annotate(f"{vb - va:+.4f}", ((va + vb) / 2, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=INK_2)
    for i, s_ in enumerate(series):
        ax.scatter(s_["values"], ys, s=95, color=SERIES[i], label=s_["name"],
                   zorder=3, edgecolor=SURFACE, linewidth=1.6)
    # Label outside the pair -- the left-hand value to the left of its dot and the
    # right-hand value to the right of its own. Centring both under the rule makes
    # them collide whenever the two measurements nearly agree.
    for y, va, vb in zip(ys, a, b):
        left, right = (0, 1) if va <= vb else (1, 0)
        vals = (va, vb)
        ax.annotate(f"{vals[left]:.4f}", (vals[left], y), textcoords="offset points",
                    xytext=(-9, 0), ha="right", va="center", fontsize=8.5,
                    color=SERIES[left])
        ax.annotate(f"{vals[right]:.4f}", (vals[right], y), textcoords="offset points",
                    xytext=(9, 0), ha="left", va="center", fontsize=8.5,
                    color=SERIES[right])
    ax.set_yticks(ys)
    ax.set_yticklabels(cats, color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(len(cats) - 0.35, -0.75)
    lo, hi = spec.get("xlim", [None, None])
    if lo is not None:
        ax.set_xlim(lo, hi)
    ax.set_xlabel(spec.get("xlabel", ""), color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="y", length=0)
    _finish(fig, path)


def _chart_hbar(spec, path):
    labels = [r["label"] for r in spec["rows"]]
    values = [r["value"] for r in spec["rows"]]
    notes = [r.get("note", "") for r in spec["rows"]]
    fig, ax = plt.subplots(figsize=(9.2, 2.4))
    ys = range(len(labels))
    bars = ax.barh(list(ys), values, height=0.52, color=SERIES[0],
                   edgecolor=SURFACE, linewidth=1.4, zorder=3)
    for b, v, note in zip(bars, values, notes):
        ax.annotate(note, (v, b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, spec.get("xmax", max(values) * 1.25))
    ax.set_xlabel(spec.get("xlabel", ""), color=INK_2, fontsize=9)
    ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.7)
    ax.grid(axis="y", visible=False)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    _finish(fig, path)


def _chart_line(spec, path):
    fig, ax = plt.subplots(figsize=(9.2, 2.6))
    best_of_all = min(p[1] for s_ in spec["series"] for p in s_["points"])
    band = spec.get("band")
    if band:
        ax.axhspan(best_of_all, best_of_all + band["width"], color=INK_3,
                   alpha=0.13, zorder=1, linewidth=0)
        # Anchored in axes fraction, not data: the empty corner is the only place
        # this caption does not cross a line.
        ax.annotate(band["label"], band.get("label_at", [0.015, 0.05]),
                    xycoords="axes fraction", fontsize=8.5, color=INK_2)
    offsets = spec.get("label_offsets") or [[0, 10]] * len(spec["series"])
    for i, s_ in enumerate(spec["series"]):
        xs = [p[0] for p in s_["points"]]
        ys = [p[1] for p in s_["points"]]
        ax.plot(xs, ys, marker="o", markersize=5.5, linewidth=2,
                color=SERIES[i], label=s_["name"],
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        bx, by = min(zip(xs, ys), key=lambda p: p[1])
        ax.annotate(f"{by:.4f}", (bx, by), textcoords="offset points",
                    xytext=offsets[i], ha="center", fontsize=8.5,
                    color=SERIES[i], fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([p[0] for p in spec["series"][0]["points"]])
    ax.set_xticklabels(spec["xticklabels"], color=INK)
    ax.minorticks_off()
    ax.set_xlabel(spec.get("xlabel", ""), color=INK_2, fontsize=9)
    ax.set_ylabel(spec.get("ylabel", ""), color=INK_2, fontsize=9)
    lo, hi = spec.get("ylim", [None, None])
    if lo is not None:
        ax.set_ylim(lo, hi)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style_axes(ax)
    _finish(fig, path)


CHARTS = {"dumbbell": _chart_dumbbell, "hbar": _chart_hbar, "line": _chart_line}


# --------------------------------------------------------------------------- slides


def _add_table(slide, table, top):
    rows, cols = len(table["rows"]) + 1, len(table["header"])
    widths = table.get("widths") or [1 / cols] * cols
    # PowerPoint grows a row when its text wraps, so a tall table can run off the
    # slide. Step the type down as the row count rises rather than trusting the
    # nominal height.
    body_pt, head_pt, row_in = (11.0, 11.5, 0.32)
    if rows >= 10:
        body_pt, head_pt, row_in = (9.5, 10.0, 0.27)
    elif rows >= 8:
        body_pt, head_pt, row_in = (10.5, 11.0, 0.29)
    height = Inches(row_in) * rows
    shape = slide.shapes.add_table(rows, cols, MARGIN, top, BODY_W, height)
    tbl = shape.table
    for j, w in enumerate(widths):
        tbl.columns[j].width = Emu(int(BODY_W * w))
    for j, h in enumerate(table["header"]):
        cell = tbl.cell(0, j)
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        for r in p.runs:
            r.font.size = Pt(head_pt)
            r.font.bold = True
            r.font.color.rgb = _rgb(INK)
    for i, row in enumerate(table["rows"], start=1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            p = cell.text_frame.paragraphs[0]
            for r in p.runs:
                r.font.size = Pt(body_pt)
                r.font.color.rgb = _rgb(INK if j == 0 else INK_2)
    return top + height


def _render_title(slide, s, _):  # noqa: D401
    tf = _box(slide, MARGIN, Inches(2.35), BODY_W, Inches(1.5))
    _run(tf.paragraphs[0], s["title"], 40, bold=True)
    if s.get("subtitle"):
        _run(tf.add_paragraph(), s["subtitle"], 19, color=INK_2, space_after=0)
    tf2 = _box(slide, MARGIN, Inches(4.35), BODY_W, Inches(2.2))
    for i, line in enumerate(s.get("lines", [])):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        _run(p, line, 14, color=INK_2, space_after=7)
    return None


def _render_body(slide, s, assets):
    tf = _box(slide, MARGIN, Inches(0.46), BODY_W, Inches(1.15))
    _run(tf.paragraphs[0], s["title"], 25, bold=True)

    top = Inches(1.52)
    size = 13.5 if s.get("chart") else 14
    for i, b in enumerate(s.get("bullets", [])):
        tfb = _box(slide, MARGIN, top, BODY_W, Inches(0.5))
        p = tfb.paragraphs[0]
        _run(p, "▪ ", size, bold=True, color=SERIES[i % 3])
        _run(p, b, size, color=INK)
        top = top + Inches(0.46 if len(b) < 62 else 0.58 if len(b) < 100 else 0.76)

    top = top + Inches(0.12)
    bottom = top
    if s.get("chart"):
        png = assets / f"{s['id']}.png"
        CHARTS[s["chart"]["type"]](s["chart"], png)
        # Fit into what is left above the footer, preserving the aspect ratio,
        # and centre it. Passing width alone lets a tall figure run off the slide.
        probe = slide.shapes.add_picture(str(png), 0, 0)
        native_w, native_h = probe.width, probe.height
        probe._element.getparent().remove(probe._element)
        avail_h = Inches(6.70) - top
        scale = min(BODY_W / native_w, avail_h / native_h)
        w, h = int(native_w * scale), int(native_h * scale)
        left = MARGIN + int((BODY_W - w) / 2)
        slide.shapes.add_picture(str(png), left, top, width=w, height=h)
        bottom = top + h
    elif s.get("table"):
        bottom = _add_table(slide, s["table"], top)

    if s.get("note"):
        tfn = _box(slide, MARGIN, Inches(6.78), BODY_W, Inches(0.6))
        _run(tfn.paragraphs[0], s["note"], 9.5, color=INK_3)
    return bottom


RENDER = {"title": _render_title, "body": _render_body}


def build(deck, out_path: Path, assets: Path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    blank = prs.slide_layouts[6]
    total = len(deck["slides"])
    for n, s in enumerate(deck["slides"], start=1):
        slide = prs.slides.add_slide(blank)
        bottom = RENDER[s.get("kind", "body")](slide, s, assets)
        if bottom and bottom > Inches(6.75):
            print(f"  ! {s['id']}: content reaches {bottom / 914400:.2f} in "
                  f"(slide is 7.50 in, footer at 6.95)")
        if n > 1:
            tf = _box(slide, SLIDE_W - MARGIN - Inches(1.6), Inches(6.95), Inches(1.6), Inches(0.3))
            tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
            _run(tf.paragraphs[0], f"{n} / {total}", 10, color=INK_3)
        if n > 1 and deck.get("footer"):
            tf = _box(slide, MARGIN, Inches(6.95), Inches(9.0), Inches(0.3))
            _run(tf.paragraphs[0], deck["footer"], 10, color=INK_3)
    prs.save(out_path)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--content", required=True, help="JSON content file (kept outside the repo)")
    ap.add_argument("--out-dir", required=True, help="directory to write the .pptx files into")
    args = ap.parse_args()

    content = json.loads(Path(args.content).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = out_dir / "_deck_assets"
    assets.mkdir(exist_ok=True)

    for name, deck in content["decks"].items():
        path = out_dir / f"{name}.pptx"
        n = build(deck, path, assets)
        print(f"wrote {path}  ({n} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
