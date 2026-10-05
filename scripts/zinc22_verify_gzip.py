"""Read each fetched ZINC-22 file to its end and report the ones that do not survive it.

A resumed transfer is spliced from two or more responses, so a wrong offset would
produce a file that exists at the right size and is still unreadable. Decompressing
to the end is the only check that catches that, and it is also what catches a
truncation the size check let through, because gzip stores its own length.

Counts the lines while it is there, so the row count per file comes from the file
rather than from an estimate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time


def verify(path):
    """Decompress one file to its end. Returns (lines, bytes, error or '')."""
    lines = raw = 0
    try:
        with gzip.open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 22)
                if not chunk:
                    break
                raw += len(chunk)
                lines += chunk.count(b"\n")
        return lines, raw, ""
    except Exception as exc:                        # noqa: BLE001 - reported, not raised
        return lines, raw, f"{type(exc).__name__}: {exc}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--report", help="JSON to write")
    ap.add_argument("--tranches", nargs="*", default=[],
                    help="restrict to these tranches; default is all of them")
    args = ap.parse_args(argv)

    targets = []
    for tranche in sorted(os.listdir(args.dest)):
        d = os.path.join(args.dest, tranche)
        if not os.path.isdir(d) or (args.tranches and tranche not in args.tranches):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".smi.gz"):
                targets.append((tranche, name, os.path.join(d, name)))

    print(f"{len(targets):,} ファイルを読み切ります", flush=True)
    bad, rows, raw_total = [], 0, 0
    t0 = time.time()
    for n, (tranche, name, path) in enumerate(targets, 1):
        lines, raw, err = verify(path)
        rows += lines
        raw_total += raw
        if err:
            bad.append({"file": f"{tranche}/{name}", "lines_read": lines, "error": err})
            print(f"  壊れている {tranche}/{name}: {err}", flush=True)
        if n % 100 == 0 or n == len(targets):
            el = time.time() - t0
            print(f"  {n:,}/{len(targets):,}  行 {rows:,}  "
                  f"展開 {raw_total / 1024**3:.1f} GiB  "
                  f"{raw_total / el / 1048576:.0f} MiB/s", flush=True)

    print(f"\n読めた {len(targets) - len(bad):,} / 壊れている {len(bad):,}")
    print(f"行 {rows:,} / 展開 {raw_total / 1024**4:.3f} TiB")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"files": len(targets), "lines": rows, "raw_bytes": raw_total,
                       "corrupt": bad}, fh, ensure_ascii=False, indent=2)
        print(f"報告 {args.report}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
