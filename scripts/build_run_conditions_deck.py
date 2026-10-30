#!/usr/bin/env python3
"""Render the run-condition table from an audit report as a .pptx, plus its TSV.

Directive `all-checkpoint-metric-order-2026-10-28-v2.md` §7 and §9: one row per run,
by modality, architecture and size, carrying the validation set, the seed actually
used, and the epochs configured against the epochs run -- as slides, with the same
numbers in a TSV beside them.

Input is the JSON written by scripts/audit_checkpoint_metric_match.py. Nothing is
measured here; this only lays out what that report holds, so the deck and the TSV
cannot disagree with each other.

One table per modality and architecture, continued over as many slides as the rows
need. The deck carries the production runs; `--all` lays out every run in the
report instead, diagnostics included.

    python scripts/build_run_conditions_deck.py \
        --audit <report.json> --out <deck.pptx> --tsv <deck-rows.tsv>
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Pt

COLUMNS = [
    ("size", 0.072),
    ("lr", 0.085),
    ("run", 0.300),
    ("検証集合", 0.215),
    ("seed", 0.125),
    ("epochs 設定", 0.100),
    ("epochs 実際", 0.103),
]
ROWS_PER_SLIDE = 13
SLIDE_W = Emu(12192000)  # 13.333 in, 16:9
SLIDE_H = Emu(6858000)
MARGIN = Emu(457200)  # 0.5 in
INK = RGBColor(0x1A, 0x1A, 0x1A)
RULE = RGBColor(0xBB, 0xBB, 0xBB)
HEAD_FILL = RGBColor(0xEE, 0xEE, 0xEE)


def cell_values(row):
    return [
        row.get("size") or "",
        _lr(row.get("learning_rate")),
        row["run"],
        row.get("検証集合") or "",
        row.get("seed_text") or "",
        _epochs(row.get("epochs_configured_from_max_steps")),
        _epochs(row.get("epochs_run")),
    ]


def _lr(value):
    if value is None:
        return ""
    return f"{float(value):g}"


def _epochs(value):
    return "" if value is None else f"{float(value):.2f}"


def add_title(slide, text):
    """One line of title. No subtitle: a deliverable carries the title alone."""
    box = slide.shapes.add_textbox(MARGIN, Emu(274320), SLIDE_W - 2 * MARGIN, Emu(457200))
    frame = box.text_frame
    frame.word_wrap = True
    para = frame.paragraphs[0]
    run = para.add_run()
    run.text = text
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = INK
    return box


def add_note(slide, lines, top):
    box = slide.shapes.add_textbox(MARGIN, top, SLIDE_W - 2 * MARGIN, Emu(914400))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(lines):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        run = para.add_run()
        run.text = line
        run.font.size = Pt(11)
        run.font.color.rgb = INK
    return box


def add_table(slide, rows, top):
    table_w = SLIDE_W - 2 * MARGIN
    height = Emu(int(320000 + 300000 * len(rows)))
    shape = slide.shapes.add_table(len(rows) + 1, len(COLUMNS), MARGIN, top,
                                   table_w, height)
    table = shape.table
    for i, (_, frac) in enumerate(COLUMNS):
        table.columns[i].width = Emu(int(int(table_w) * frac))

    for i, (name, _) in enumerate(COLUMNS):
        cell = table.cell(0, i)
        cell.text = name
        para = cell.text_frame.paragraphs[0]
        para.runs[0].font.size = Pt(10)
        para.runs[0].font.bold = True
        para.runs[0].font.color.rgb = INK
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEAD_FILL

    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(cell_values(row)):
            cell = table.cell(r, c)
            cell.text = str(value)
            para = cell.text_frame.paragraphs[0]
            if para.runs:
                para.runs[0].font.size = Pt(9)
                para.runs[0].font.color.rgb = INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    return shape


def blank_slide(prs):
    # Layout 6 is the blank one in the default template; every element is placed
    # explicitly so no placeholder text can survive into the deck.
    return prs.slides.add_slide(prs.slide_layouts[6])


def build(rows, out_path, excluded, source):
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    groups = {}
    for row in rows:
        groups.setdefault((row["modality"], row["arch"]), []).append(row)

    lead = blank_slide(prs)
    add_title(lead, "run の条件（検証集合・seed・エポック数）")
    counts = "、".join(f"{m}/{a} {len(v)} 本" for (m, a), v in sorted(groups.items()))
    add_note(lead, [
        f"対象 {len(rows)} 本。{counts}",
        f"診断・スモーク・スループット計測の {excluded} 本は除いた。"
        "除外した run も TSV と JSON には入っている。",
        "エポック数の「設定」は max_steps ÷ 1 エポックあたり step、"
        "「実際」は完走 step ÷ 1 エポックあたり step。",
        "seed は run が実際に使った値。config の指定と違う run は括弧で併記した。",
        f"出どころ: {source}",
    ], Emu(1097280))

    for (modality, arch), group in sorted(groups.items()):
        group.sort(key=lambda r: (str(r.get("size")), r["run"]))
        chunks = [group[i:i + ROWS_PER_SLIDE]
                  for i in range(0, len(group), ROWS_PER_SLIDE)]
        for n, chunk in enumerate(chunks, start=1):
            slide = blank_slide(prs)
            suffix = f"（{n}/{len(chunks)}）" if len(chunks) > 1 else ""
            add_title(slide, f"{modality} / {arch}{suffix}")
            add_table(slide, chunk, Emu(960120))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    prs.save(out_path)
    return len(prs.slides)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True, help="the .pptx to write")
    ap.add_argument("--tsv", help="the same rows as a TSV")
    ap.add_argument("--all", action="store_true",
                    help="lay out every run, diagnostics included")
    args = ap.parse_args(argv)

    with open(args.audit) as fh:
        audit = json.load(fh)

    # The audit script's own helpers produce these two columns; importing them
    # keeps one definition rather than a second one that can drift.
    from audit_checkpoint_metric_match import (
        eval_set_description,
        is_production,
        seed_description,
    )

    rows = []
    excluded = 0
    for row in audit["runs"]:
        if not args.all and not is_production(row):
            excluded += 1
            continue
        row = dict(row)
        row["検証集合"] = eval_set_description(row)
        row["seed_text"] = seed_description(row)
        rows.append(row)

    source = os.path.basename(args.audit)
    slides = build(rows, args.out, excluded, source)
    print(f"wrote {args.out} ({slides} slides, {len(rows)} runs, {excluded} excluded)")

    if args.tsv:
        header = ["modality", "arch"] + [name for name, _ in COLUMNS] + [
            "max_steps_設定", "steps_完走", "train_rows", "global_batch",
            "steps_per_epoch", "job_id"]
        with open(args.tsv, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            for row in rows:
                writer.writerow([row["modality"], row["arch"]] + cell_values(row) + [
                    row.get("max_steps") or "", row.get("steps_run") or "",
                    row.get("train_rows") or "", row.get("global_batch") or "",
                    "" if row.get("steps_per_epoch") is None
                    else f"{row['steps_per_epoch']:.2f}",
                    row.get("job_id") or "",
                ])
        print(f"wrote {args.tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
