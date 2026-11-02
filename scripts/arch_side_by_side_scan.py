"""Find every table that puts two architectures side by side for one modality.

Written because a hand-picked list cannot answer "list all of them". The first
attempt at this inventory was assembled by choosing candidate files and scanning
those, and it missed a dozen tables -- all of them cross-modality tables that carry
a row for the modality in question and a column per architecture. This scans
everything and decides by a stated rule instead.

The rule is that the architecture has to be an *axis* of the table, one of:

  header   a column whose header names an architecture ("| | BERT | GPT-2 |")
  key      a column named arch / アーキテクチャ, with both architectures among its values
  row      a row whose first cell names an architecture

That is what "並べている" means, and it is also what separates a real hit from a
file listing or a decision log, where the architecture names appear inside prose
cells and no column belongs to either one. Both architectures must be present; one
alone is not a comparison.

The subject -- what is being put side by side -- is inferred from the header and the
first column by the keyword table below, because that is the part of the question a
reader actually needs: a loss table and a seed inventory raise different problems.
``subject = その他`` means the rule found an axis but no keyword, and those rows are
meant to be read, not trusted.

Markdown tables and TSVs are read. HTML is not parsed: the pages in this project are
generated from the TSVs, so the TSV row carries them, and the generator is named in
``note`` instead of a second, weaker parser.

Read-only. Paths are arguments: this repository is public.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ARCH = {
    "bert": re.compile(r"(?<![a-z])bert(?![a-z])", re.I),
    "gpt2": re.compile(r"gpt-?2", re.I),
}
ARCH_KEY = re.compile(r"^\s*(arch|architecture|アーキテクチャ|経路)\s*$", re.I)
# For a TSV column header: the cell is the architecture and nothing else.
ARCH_NAME = {
    "bert": re.compile(r"bert(-\w+)?", re.I),
    "gpt2": re.compile(r"gpt-?2(-\w+)?", re.I),
}

# What the table is putting side by side. Checked in order; the first match wins,
# so the more specific subjects come first.
SUBJECTS = [
    ("下流", r"RMSE|ROC-?AUC|溶解度|hERG|AqSolDB|Spearman|recall@|下流"
             r"|trained_fold|untrained|fingerprint"),
    ("事前学習 loss", r"loss|損失|採用値|best_val|perplexity|eval_loss"
                      r"|adopted_value|best_metric"),
    ("エポック・トークン", r"epoch|エポック|トークン|token"),
    ("パラメータ数", r"params|パラメータ|非埋め込み"),
    ("学習率", r"学習率|learning[_ ]rate|(?<![a-z_])lr(?![a-z_])|_lr\b|\blr_"
               r"|min_lr|スケジュール|schedule"),
    ("seed", r"\bseed\b|シード"),
    # Not a bare "valid": it is the name of a split, and the loss-curve TSV's own
    # split column is full of it, which read the curves as a table about validation
    # sets instead of about loss.
    ("検証集合・評価間隔", r"検証集合|eval_interval|save_interval|評価間隔|validation"),
    ("checkpoint・保存形式", r"checkpoint|保存|採用 step|ckpt"),
    ("計算資源", r"GPU-h|GPU 時間|計算量|PF-?day|バッチ|batch"),
    ("件数・在処", r"本数|件数|在処|パス|ファイル|ツリー|\brun\b"),
]
# Columns that say *which row this is*, not what the table is about. A TSV headed
# "arch size lr seed step split value" is not a table about learning rates, and
# reading "lr" as the subject put the loss curves under 学習率.
KEY_COLUMNS = {
    "arch", "architecture", "size", "lr", "seed", "step", "steps", "iter", "run",
    "source", "job", "modality", "task", "checkpoint", "tree", "set", "n", "split",
    "hidden", "layers", "eval", "file", "name", "date", "state", "note",
}
COLUMNS = ["file", "location", "format", "modality_scope", "axis", "subject",
           "rows", "header", "note"]


def subject_text(header: list[str]) -> str:
    """The header with its key columns dropped, for subject inference."""
    return " ".join(c for c in header if c.strip().lower() not in KEY_COLUMNS)


def subject_body(header: list[str], rows, limit: int = 200) -> str:
    """The rows with their key columns dropped.

    The same reason the header drops them: a ``source`` column full of
    ``checkpoint-45000/trainer_state.json`` read the loss curves as a table about
    checkpoints, and a ``split`` column full of ``valid`` read them as one about
    validation sets. What a row says about the subject is in its value columns.
    """
    keep = [i for i, c in enumerate(header) if c.strip().lower() not in KEY_COLUMNS]
    return "\n".join(
        "\t".join(r[i] for i in keep if len(r) > i) for r in rows[:limit])


# The two subjects the question is actually about: these are the values that must
# not be read across architectures. They are searched in every piece of evidence
# before anything else, because a downstream table headed
# "学習済み | 未学習（seed 3 本）| 差" otherwise comes out as a table about seeds --
# the word is in its header, while "RMSE" is only in the heading above it.
PRIORITY = 2


def subject_of(*texts: str) -> str:
    """The first subject whose keywords appear, looking at each text in turn.

    The caller passes the narrowest evidence first -- the table's own header, then
    the heading it sits under, then its cells -- because a table headed only with
    size names says what it is about nowhere but in its cells, while a loss table
    that mentions epochs in a footnote should still come out as a loss table. The
    two priority subjects are exempt from that order.
    """
    for name, pattern in SUBJECTS[:PRIORITY]:
        if any(re.search(pattern, t, re.I) for t in texts):
            return name
    for text in texts:
        for name, pattern in SUBJECTS[PRIORITY:]:
            if re.search(pattern, text, re.I):
                return name
    return "その他"


def split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"[\s|:\-]+", line.strip())) and "-" in line


IN_ONE_CELL = "セル内（列でも行でもない）"


def both_in_one_cell(cells) -> bool:
    """A cell that puts a value against each architecture.

    Not an axis, but still a place where the two are set next to each other:
    corpus-size's epochs cell holds "BERT: 96.26 … ｜ GPT-2: 29.99 …" in a table
    whose rows are modalities.

    A number has to follow each name. Without that, "compounds の BERT と GPT-2 の
    曲線" in a file listing counts as a comparison, and the listings outnumber the
    comparisons.
    """
    for cell in cells:
        if all(re.search(rx.pattern + r"[^0-9]{0,40}\d", cell, re.I)
               for rx in ARCH.values()):
            return True
    return False


def axis_of(block: list[str]) -> str | None:
    """Which way the architecture runs across the table, or None if it does not."""
    rows = [split_row(line) for line in block if not is_separator(line)]
    if not rows:
        return None
    header = rows[0]
    # header: two different columns, each naming one architecture
    named = {a for cell in header for a, rx in ARCH.items() if rx.search(cell)}
    if len(named) == 2:
        return "列（見出しが arch）"
    # key: a column named arch, with both architectures among its values
    for i, cell in enumerate(header):
        if ARCH_KEY.match(cell):
            values = " ".join(r[i] for r in rows[1:] if len(r) > i)
            if all(rx.search(values) for rx in ARCH.values()):
                return f"列（{cell}）"
    # row: first cells naming an architecture, both present
    firsts = " ".join(r[0] for r in rows[1:] if r)
    if all(rx.search(firsts) for rx in ARCH.values()):
        return "行（先頭セルが arch）"
    if both_in_one_cell(c for r in rows for c in r):
        return IN_ONE_CELL
    return None


HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def markdown_tables(path: Path):
    """Each table, with the heading it sits under.

    The heading is carried because that is often the only place the subject is
    written: "### 溶解度（AqSolDB、9,982 分子、RMSE、小さいほど良い）" over a table whose
    own header is just 学習済み / 未学習 / 差.
    """
    lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    i, heading = 0, ""
    while i < len(lines):
        m = HEADING.match(lines[i])
        if m:
            heading = m.group(1)
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            yield i + 1, lines[i:j], heading
            i = j
        else:
            i += 1


def scan_markdown(path: Path, modality: re.Pattern, rel: str, rows: list):
    whole_file_is_modality = bool(modality.search(path.name))
    for start, block, heading in markdown_tables(path):
        axis = axis_of(block)
        if axis is None:
            continue
        body = "\n".join(block)
        in_table = bool(modality.search(body))
        if not (in_table or whole_file_is_modality):
            continue
        header = next((line for line in block if not is_separator(line)), "")
        first_col = " ".join(split_row(line)[0] for line in block
                             if not is_separator(line))
        rows.append({
            "file": rel, "location": f"L{start}", "format": "markdown 表",
            "modality_scope": "この表の中" if in_table else "この文書全体",
            # The table's own header first, then the heading it sits under, then its
            # cells. The document's file name is never used: one markdown document
            # carries tables on many subjects.
            "axis": axis,
            "subject": subject_of(header + " " + first_col, heading, body),
            "rows": str(len(block)), "header": header.strip()[:120],
            "note": "",
        })


def scan_tsv(path: Path, modality: re.Pattern, rel: str, rows: list):
    try:
        with open(path, encoding="utf-8-sig", errors="ignore", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader, None)
            if not header:
                return
            data = [r for r in reader]
    except OSError:
        return
    if not data:
        return
    # the arch axis in a TSV is a column, by header name or by its values
    # A header cell has to *be* an architecture name, not merely contain one. A TSV
    # written without a header row puts data on the first line, and a job name like
    # "ds45k-solubility-bert-small" would otherwise be read as an architecture column.
    idx = next((i for i, c in enumerate(header) if ARCH_KEY.match(c)), None)
    if idx is None:
        idx = next((i for i, c in enumerate(header)
                    if any(rx.fullmatch(c.strip()) for rx in ARCH_NAME.values())), None)
    if idx is None:
        # No architecture column. It can still be a cell holding both, as
        # corpus-size's epochs cell is.
        hits = [r for r in data if modality.search("\t".join(r))
                and both_in_one_cell(r)]
        if not hits:
            return
        rows.append({
            "file": rel, "location": f"{hits[0][0]} 行", "format": "TSV",
            "modality_scope": "この表の中", "axis": IN_ONE_CELL,
            "subject": subject_of(subject_text(header), subject_body(header, hits),
                                  Path(rel).stem.replace("-", " ")),
            "rows": f"{len(hits)} 行（1 セルに両方）",
            "header": "\t".join(header)[:120], "note": "",
        })
        return
    values = " ".join(r[idx] for r in data if len(r) > idx)
    if not all(rx.search(values) for rx in ARCH.values()):
        return
    hits = [r for r in data if modality.search("\t".join(r))]
    scope = "この表の中"
    if not hits:
        if not modality.search(path.name):
            return
        hits, scope = data, "この文書全体"
    counts = {}
    for a, rx in ARCH.items():
        counts[a] = sum(1 for r in hits if len(r) > idx and rx.search(r[idx]))
    rows.append({
        "file": rel, "location": f"{header[idx]} 列", "format": "TSV",
        "modality_scope": scope, "axis": f"列（{header[idx]}）",
        # The header first, then the rows: a TSV header is terse and in English
        # ("metric", "adopted_value"), and what the table is about is often only
        # legible in its cells ("eval_loss_mask", "RMSE").
        # The header, then the rows, then the file name. The name is evidence for a
        # TSV because the file *is* one table -- "compounds-loss-curves" says what it
        # holds where the columns ("split", "value") do not. A markdown document
        # carries many tables, so its name is never used that way.
        "subject": subject_of(subject_text(header), subject_body(header, hits),
                              Path(rel).stem.replace("-", " ")),
        "rows": f"bert {counts['bert']} / gpt2 {counts['gpt2']}",
        "header": "\t".join(header)[:120], "note": "",
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="the directory of deliverables to scan")
    ap.add_argument("--modality", default=r"compounds|化合物|cmp-|溶解度|hERG|AqSolDB",
                    help="regex that marks a row or a file as belonging to the modality")
    ap.add_argument("--skip", action="append", default=["_archive", "_scratch"],
                    help="path fragments to leave out (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    modality = re.compile(args.modality, re.I)
    rows: list[dict] = []
    out = args.out.resolve()
    for path in sorted(args.root.rglob("*")):
        rel = str(path.relative_to(args.root))
        if any(s in rel for s in args.skip) or not path.is_file():
            continue
        # Not this table. It names the files it found, both architectures among
        # them, so scanning it would list itself on every run.
        if path.resolve() == out:
            continue
        if path.suffix == ".md":
            scan_markdown(path, modality, rel, rows)
        elif path.suffix == ".tsv":
            scan_tsv(path, modality, rel, rows)

    rows.sort(key=lambda r: (r["subject"], r["file"], r["location"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # BOM so the file is not taken for binary when it is shared on Slack.
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    by_subject: dict[str, int] = {}
    for r in rows:
        by_subject[r["subject"]] = by_subject.get(r["subject"], 0) + 1
    for subject, n in sorted(by_subject.items(), key=lambda x: -x[1]):
        print(f"  {subject:<22}{n:>4}")
    print(f"\n{len(rows)} 行 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
