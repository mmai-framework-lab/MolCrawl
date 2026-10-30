"""Convert a Marp deck to .pptx, reusing this repo's slide renderer.

marp-cli needs Node, which is not installed on this cluster, and the deck's
source of truth is the Marp markdown -- so the markdown is parsed here and handed
to the same layout code that build_review12_deck.py uses. One renderer, one set of
layout rules, no second place for them to drift.

Supported per slide: an h1/h2 heading, "- " bullets, a GitHub pipe table, an
"![](path)" image, and Marp's "<!-- footer: ... -->" comment, which becomes the
footnote line.

Inline markup is NOT carried over. The renderer writes one run per bullet, so a
"**...**" would reach the slide as literal asterisks rather than as bold; both it
and `code` ticks are stripped instead of being faked.

    python scripts/marp_to_pptx.py --md <deck.md> --out-dir <dir> [--base-dir <dir>]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from build_review12_deck import build

FOOTER_RE = re.compile(r"<!--\s*_?footer:\s*([\s\S]*?)-->")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def clean(text: str) -> str:
    """Drop the inline markup the renderer has no run style for.

    "==...==" is kept: the renderer reads it as the accent for what moved in the
    period being reported.
    """
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text).replace("`", "").strip()


def parse_table(lines):
    rows = [[clean(c) for c in line.strip().strip("|").split("|")] for line in lines]
    header, body = rows[0], rows[2:]          # rows[1] is the |---|---| rule
    cols = len(header)
    # Give each column room in proportion to its widest cell, so a column of
    # four-digit numbers does not get the same width as a sentence.
    span = [max((len(r[i]) for r in [header] + body if i < len(r)), default=1)
            for i in range(cols)]
    span = [max(4, min(s, 48)) for s in span]
    total = sum(span)
    return {"header": header, "rows": body, "widths": [s / total for s in span]}


def parse_slide(md: str, base: Path, index: int):
    notes = FOOTER_RE.findall(md)
    md = FOOTER_RE.sub("", md)

    slide = {"id": f"s{index}", "bullets": []}
    if notes:
        slide["note"] = " ／ ".join(n.strip() for n in notes)

    lines = md.split("\n")
    table_buf, free, free_after = [], [], False
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|"):
            table_buf.append(line)
            continue
        if table_buf:
            slide.setdefault("tables", []).append(parse_table(table_buf))
            table_buf = []
        if line.startswith("# "):
            slide["kind"], slide["title"] = "title", clean(line[2:])
        elif line.startswith("## "):
            slide["title"] = clean(line[3:])
        elif line.startswith("- "):
            slide["bullets"].append(clean(line[2:]))
            free_after = True
        elif IMG_RE.search(line):
            src = IMG_RE.search(line).group(1)
            slide["chart"] = {"type": "image", "path": str((base / src).resolve())}
        elif line.strip():
            free.append(clean(line))
    if table_buf:
        slide.setdefault("tables", []).append(parse_table(table_buf))

    if slide.get("tables"):
        # A slide may carry several tables; the renderer stacks them in order.
        slide["table"] = slide["tables"][0]
    if slide.get("kind") == "title":
        slide["lines"] = free
    elif free:
        # Keep source order: a bare paragraph before the bullets is a lead-in, one
        # after them introduces the table that follows.
        slide["bullets"] = (slide["bullets"] + free) if free_after else (free + slide["bullets"])
    return slide


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-dir", default=None, help="resolves image paths (default: the md's dir)")
    ap.add_argument("--footer", default="")
    args = ap.parse_args()

    md_path = Path(args.md)
    base = Path(args.base_dir) if args.base_dir else md_path.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = md_path.read_text(encoding="utf-8").lstrip("﻿")
    parts = re.split(r"\n---\n", text)
    slides = [parse_slide(p, base, i) for i, p in enumerate(parts[1:], start=1)]

    deck = {"footer": args.footer, "slides": slides}
    assets = out_dir / "_deck_assets"
    assets.mkdir(exist_ok=True)
    out = out_dir / (md_path.stem + ".pptx")
    n = build(deck, out, assets)
    print(f"wrote {out}  ({n} slides)")
    for s in slides:
        bits = []
        if s.get("chart"):
            bits.append("図")
        for tb in s.get("tables", []):
            bits.append(f"表 {len(tb['rows'])} 行")
        bits.append(f"箇条書き {len(s.get('bullets', []))}")
        print(f"  {s['id']:4s} {s.get('title','')[:40]:42s} {' / '.join(bits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
