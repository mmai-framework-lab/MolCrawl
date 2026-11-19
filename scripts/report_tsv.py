"""Write a report TSV that declares the column convention in its name and contents.

``docs/07-reports/report_tsv_columns.md`` is the convention, version 1. Its §10
asks that a script writing a new TSV write the ``rt1-`` name and the provenance
block from the start rather than leaving a later pass to convert the file, so
this module holds the part every such script would otherwise repeat: the name
check, the block, and the rules a writer can break without noticing.

    from report_tsv import stamp, write_tsv

    write_tsv(out, ["size", "step", "eval_loss_mask", "source"], rows, {
        "model": "compounds bert small",
        "source": "<run>/checkpoint-45000/trainer_state.json",
        "generated": stamp("scripts/example.py"),
    })

What is checked here, with the rule number from §7 of the convention:

    the name begins with ``rt1-``, and the body declares 1   §2, §3, R13
    ``model``, ``source`` and ``generated`` are given        R14
    no column name repeats                                   R4
    a column name holds only ASCII alphanumerics, ``_@.``    R3
    a ``value`` column comes with a ``metric`` column        R12
    no BOM, and LF line endings                              R5, R6
    a missing value is an empty cell, not ``NaN`` or ``-``   R10
    every row has as many fields as the header               R2

What is not checked: whether the column names are the ones §6 asks for, whether
a value column exists, and whether the table should have been form B rather than
form A. Those are judgements about the table, and §9 is where they are made.
"""

from __future__ import annotations

import csv
import datetime as _dt
import re
from pathlib import Path

VERSION = 1
PREFIX = f"rt{VERSION}-"
BANNER = f"#! report-tsv {VERSION}"

REQUIRED_KEYS = ("model", "source", "generated")

# The order §3's example writes them in. A key not listed keeps the caller's order
# and follows these.
KEY_ORDER = ("model", "run", "checkpoint", "source", "job", "generated", "note")

NAME_RE = re.compile(r"^[A-Za-z0-9_@.]+$")

# What R10 forbids in a cell: a missing value is written as nothing at all, because
# every one of these reads as a string to one reader and as a number to another.
MISSING_MARKERS = {"nan", "NaN", "NA", "N/A", "null", "NULL", "None", "-", "--"}

JST = _dt.timezone(_dt.timedelta(hours=9))


def stamp(script: str, now: _dt.datetime | None = None) -> str:
    """The ``generated`` value: ``<ISO8601+09:00> by <script>``."""
    moment = now or _dt.datetime.now(JST)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=JST)
    return f"{moment.isoformat(timespec='seconds')} by {script}"


def check_name(path: Path | str) -> Path:
    """The file name states the convention (§2). Refuse rather than write a file
    whose name does not say what is inside it."""
    out = Path(path)
    if not out.name.startswith(PREFIX):
        raise SystemExit(
            f"--out must name a file whose base name begins with {PREFIX!r} "
            f"(the column convention, docs/07-reports/report_tsv_columns.md §2); "
            f"got {out.name!r}"
        )
    return out


def _check_columns(columns: list[str]) -> None:
    if not columns:
        raise SystemExit("a TSV needs at least one column")
    seen = [c for c in columns if columns.count(c) > 1]
    if seen:
        raise SystemExit(f"duplicate column name (R4): {sorted(set(seen))}")
    bad = [c for c in columns if not NAME_RE.match(c)]
    if bad:
        raise SystemExit(
            f"a column name holds only ASCII alphanumerics, '_', '@' and '.' (R3): {bad}"
        )
    if "value" in columns and "metric" not in columns:
        raise SystemExit(
            "a 'value' column comes with a 'metric' column (R12). A table whose "
            "metric does not change from row to row names the value column after "
            "the metric instead (form B, §9)"
        )


def _check_provenance(provenance: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if not str(provenance.get(k, "")).strip()]
    if missing:
        raise SystemExit(f"the provenance needs {list(REQUIRED_KEYS)} (R14); missing {missing}")
    for key, value in provenance.items():
        if not NAME_RE.match(key):
            raise SystemExit(f"provenance key {key!r} is not a bare ASCII name")
        for one in value if isinstance(value, (list, tuple)) else [value]:
            if "\n" in str(one) or "\r" in str(one):
                raise SystemExit(f"provenance value for {key!r} spans lines")
            if str(one).strip().lower() in {"unknown", "不明"}:
                raise SystemExit(f"provenance {key!r} is {one!r}; R15 asks for the real value")


def _cell(value) -> str:
    """One field. ``None`` is the empty cell R10 asks for."""
    if value is None:
        return ""
    text = str(value)
    if text in MISSING_MARKERS:
        raise SystemExit(
            f"a missing value is an empty cell, not {text!r} (R10); pass None"
        )
    if "\t" in text or "\n" in text or "\r" in text:
        raise SystemExit(f"a field holds a tab or a line break: {text!r}")
    return text


def provenance_block(provenance: dict) -> list[str]:
    """The banner and the ``# key: value`` lines, in §3's order.

    An optional key the caller had no value for -- the Slurm job id outside a job,
    say -- is left out rather than written as ``None``. R15 is about not claiming
    a value is unknown, and a line that is not there claims nothing.
    """
    provenance = {k: v for k, v in provenance.items()
                  if k in REQUIRED_KEYS or str(v if v is not None else "").strip()}
    _check_provenance(provenance)
    keys = [k for k in KEY_ORDER if k in provenance]
    keys += [k for k in provenance if k not in KEY_ORDER]
    lines = [BANNER]
    for key in keys:
        value = provenance[key]
        # Several lines under one key -- two notes, say -- are a list. §3 allows any
        # number of `# note:` lines and a dict cannot hold the key twice.
        for one in value if isinstance(value, (list, tuple)) else [value]:
            lines.append(f"# {key}: {one}")
    return lines


def write_tsv(path: Path | str, columns: list[str], rows, provenance: dict) -> int:
    """Write the file and return how many data rows it holds.

    ``rows`` are dicts keyed by the column names. A column a row does not carry
    is an empty cell.
    """
    out = check_name(path)
    columns = list(columns)
    _check_columns(columns)
    block = provenance_block(provenance)

    data = []
    for i, row in enumerate(rows):
        unknown = [k for k in row if k not in columns]
        if unknown:
            raise SystemExit(f"row {i} has fields the header does not (R2): {unknown}")
        data.append({c: _cell(row.get(c)) for c in columns})

    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8 with no BOM (R5) and LF alone (R6). csv writes CRLF unless told.
    with open(out, "w", newline="", encoding="utf-8") as fh:
        for line in block:
            fh.write(line + "\n")
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    return len(data)
