"""Account for the bases a chunk length drops, line by line.

Rebuilding the BERT data at 1,024 turned out to use 1.37% fewer bases than the
512 build. The remainder of a full Phase 2 line explains only 0.195% of that
(261,120 / 1022 = 255 remainder 510), so something else accounts for the other
1.17%.

The suspicion is short lines. A line shorter than the chunk yields no windows at
all, and contigs do not end on a line boundary, so every contig contributes a
tail line. At 510 a 700-base tail still yields one window; at 1,022 it yields
none. This walks the raw files and settles it:

    full lines        RAW_LINE_LEN exactly -- loses only the remainder
    short lines       a contig tail -- loses the remainder, or everything
    unusable lines    shorter than the chunk -- loses all of it

The three have to add up to the measured shortfall. Reports per accession and in
total, for whatever chunk sizes are given, so 510 and 1022 can be compared on
the same lines.
"""

import argparse
import os
from collections import Counter

RAW_LINE_LEN = 261_120


def scan(raw_dir, chunks):
    """Per chunk size: bases kept, bases lost, and where the loss came from."""
    stats = {c: Counter() for c in chunks}
    lines = Counter()
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".raw"):
            continue
        with open(os.path.join(raw_dir, name), "rb") as fh:
            for raw in fh:
                n = len(raw.rstrip(b"\n"))
                if not n:
                    continue
                lines["count"] += 1
                lines["bases"] += n
                lines["full" if n == RAW_LINE_LEN else "short"] += 1
                for c in chunks:
                    kept = (n // c) * c
                    s = stats[c]
                    s["kept"] += kept
                    s["lost"] += n - kept
                    if n < c:
                        s["unusable_lines"] += 1
                        s["lost_unusable"] += n
                    elif n == RAW_LINE_LEN:
                        s["lost_full_remainder"] += n - kept
                    else:
                        s["lost_short_remainder"] += n - kept
    return lines, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", help="a subset's raw_files/ directory")
    ap.add_argument("--chunks", default="510,1022")
    args = ap.parse_args()
    chunks = [int(x) for x in args.chunks.split(",")]

    lines, stats = scan(args.raw_dir, chunks)
    total = lines["bases"]
    print(f"  lines: {lines['count']:,}  "
          f"({lines['full']:,} full, {lines['short']:,} short)")
    print(f"  bases: {total:,}\n")
    print(f"  {'chunk':>6}{'windows':>14}{'kept':>16}{'lost':>14}{'lost %':>9}"
          f"{'  breakdown of the loss'}")
    for c in chunks:
        s = stats[c]
        pct = s["lost"] / total * 100 if total else 0
        print(f"  {c:>6}{s['kept'] // c:>14,}{s['kept']:>16,}{s['lost']:>14,}{pct:>8.3f}%"
              f"   full-line remainder {s['lost_full_remainder']:,}"
              f" / short-line remainder {s['lost_short_remainder']:,}"
              f" / lines too short to use {s['lost_unusable']:,}"
              f" ({s['unusable_lines']:,} lines)")
    if len(chunks) == 2:
        a, b = chunks
        ka, kb = stats[a]["kept"], stats[b]["kept"]
        if ka:
            print(f"\n  bases at {b} relative to {a}: {kb / ka:.4f}  "
                  f"(shortfall {(1 - kb / ka) * 100:.3f}%)")


if __name__ == "__main__":
    main()
