#!/usr/bin/env python3
"""Check a report TSV against the required rules of the column convention.

The nine rules a file must keep are listed in docs/07-reports/report_tsv_columns.md: a
file that breaks one of them either cannot be read by a machine or can be read wrongly.
This checks those nine and nothing else, so a pass here is not a statement about the
recommended rules.

    python scripts/verify_report_tsv.py <tsv>...

Exit status is 1 when any file breaks a required rule. The official checker,
model-web/tools/verify-rt1.mjs, lives in an unmerged branch and cannot read a native rt1
file; this one reads what the convention document says.
"""

from __future__ import annotations

import csv
import io
import os
import re
import sys

SENTINELS = {"nan", "NaN", "NAN", "NA", "N/A", "None", "none", "null", "NULL", "-", "--"}
HEADER = re.compile(r"^#! report-tsv (\d+)$")
REQUIRED_KEYS = ("model", "source", "generated")


def check(path):
    problems = []
    raw = open(path, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("R5: the file starts with a BOM")
        raw = raw[3:]
    if b"\r" in raw:
        problems.append("R6: the file contains CR, so the line endings are not LF only")
    lines = raw.decode("utf-8").split("\n")

    version = None
    body, provenance = [], {}
    for line in lines:
        got = HEADER.match(line)
        if got:
            version = int(got.group(1))
            continue
        if line.startswith("#"):
            key = line.lstrip("#").strip().split(":", 1)
            if len(key) == 2 and re.fullmatch(r"[a-z_]+", key[0].strip()):
                provenance.setdefault(key[0].strip(), []).append(key[1].strip())
            continue
        if line.strip():
            body.append(line)

    if version is None:
        problems.append("R1/R13: no '#! report-tsv <n>' line")
    name = os.path.basename(path)
    in_name = re.match(r"^rt(\d+)-", name)
    if in_name and version is not None and int(in_name.group(1)) != version:
        problems.append(f"R13: the name says rt{in_name.group(1)} and the contents say {version}")
    if not in_name:
        problems.append(f"R13: the file name {name} does not begin with rt<n>-")

    for key in REQUIRED_KEYS:
        if key not in provenance:
            problems.append(f"R14: the provenance has no '{key}'")
    stamp = (provenance.get("generated") or [""])[0]
    if stamp and " by " not in stamp:
        problems.append("R14: 'generated' is not '<ISO8601+09:00> by <script>'")

    if not body:
        problems.append("R1: there is no header row")
        return problems
    rows = list(csv.reader(io.StringIO("\n".join(body)), delimiter="\t"))
    header = rows[0]
    if len(header) != len(set(header)):
        problems.append("R4: duplicate column names")
    wrong = [i for i, row in enumerate(rows[1:], start=2) if len(row) != len(header)]
    if wrong:
        problems.append(f"R2: {len(wrong)} row(s) do not have {len(header)} fields "
                        f"(first at data row {wrong[0]})")
    hits = [(i, c) for i, row in enumerate(rows[1:], start=2) for c in row if c.strip() in SENTINELS]
    if hits:
        problems.append(f"R10: {len(hits)} cell(s) hold a missing-value word rather than an "
                        f"empty cell (first at data row {hits[0][0]}: {hits[0][1]!r})")
    if "value" in header and "metric" not in header:
        problems.append("R12: a 'value' column without a 'metric' column")
    if not problems:
        print(f"{path}: {len(rows) - 1} data rows, {len(header)} columns, all 9 required rules kept")
    return problems


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    bad = 0
    for path in argv:
        problems = check(path)
        if problems:
            bad += 1
            print(path)
            for p in problems:
                print("   ", p)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
