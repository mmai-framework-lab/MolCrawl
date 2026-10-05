"""Check the fetched ZINC-22 tree against the list it was supposed to come from.

The acquisition record says what each attempt did, but a record can only report the
attempts that happened. This compares three things that should agree and names every
file where they do not:

  the inventory   -- what the list said should exist
  the record      -- what the fetch reports it did
  the tree        -- what is actually on disk, with its size

A file counted as taken but missing from disk, a file on disk that the list never
named, and a size that disagrees with the listed one are each reported separately,
because they fail for different reasons and only the first two mean the corpus is
incomplete. Sizes in the inventory are Apache's rounded column, so a disagreement is
only reported outside a tolerance.

Exits non-zero when anything is missing, so a workflow can stop on it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

SIZE_TOLERANCE = 0.12


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--record", nargs="+", required=True,
                    help="one or more fetch records; later ones win for a repeated file")
    ap.add_argument("--report", help="JSON to write")
    ap.add_argument("--show", type=int, default=20, help="how many names to print per category")
    args = ap.parse_args(argv)

    listed = {}
    with open(args.inventory, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            listed[(row["tranche"], row["file"])] = int(row["size_bytes"])

    status = {}
    for path in args.record:
        if not os.path.exists(path):
            print(f"記録が無い: {path}")
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                status[(row["tranche"], row["file"])] = row["status"]

    on_disk = {}
    for tranche in sorted(os.listdir(args.dest)):
        d = os.path.join(args.dest, tranche)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".part"):
                continue
            on_disk[(tranche, name)] = os.path.getsize(os.path.join(d, name))

    missing, unlisted, size_off, never_tried = [], [], [], []
    for key, want in listed.items():
        if key not in on_disk:
            missing.append(key)
            if key not in status:
                never_tried.append(key)
        elif want and abs(on_disk[key] - want) > want * SIZE_TOLERANCE:
            size_off.append((key, want, on_disk[key]))
    for key in on_disk:
        if key not in listed:
            unlisted.append(key)

    got_bytes = sum(on_disk[k] for k in on_disk if k in listed)
    want_bytes = sum(listed.values())

    def show(label, items, fmt=lambda x: f"{x[0]}/{x[1]}"):
        print(f"\n{label}: {len(items):,}")
        for item in items[: args.show]:
            print(f"  {fmt(item)}")
        if len(items) > args.show:
            print(f"  ... 他 {len(items) - args.show:,} 件")

    print(f"一覧      {len(listed):,} ファイル / {want_bytes / 1024**4:.2f} TiB")
    print(f"ディスク上 {len(on_disk):,} ファイル / {got_bytes / 1024**4:.2f} TiB "
          f"（一覧に載るものだけ）")
    print(f"記録       {len(status):,} 行")
    show("一覧にあってディスクに無い", missing)
    show("そのうち一度も試行されていない", never_tried)
    show("ディスクにあって一覧に無い", unlisted)
    show("サイズが一覧と合わない", size_off,
         lambda x: f"{x[0][0]}/{x[0][1]} 一覧 {x[1]:,} 実際 {x[2]:,}")

    complete = not missing
    print(f"\n{'完全' if complete else '不完全'}: "
          f"{len(listed) - len(missing):,}/{len(listed):,} ファイル")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({
                "listed_files": len(listed),
                "listed_bytes": want_bytes,
                "on_disk_files": len(on_disk),
                "on_disk_bytes": got_bytes,
                "record_rows": len(status),
                "missing": [f"{t}/{f}" for t, f in missing],
                "never_attempted": [f"{t}/{f}" for t, f in never_tried],
                "unlisted": [f"{t}/{f}" for t, f in unlisted],
                "size_mismatch": [{"file": f"{k[0]}/{k[1]}", "listed": w, "actual": a}
                                  for k, w, a in size_off],
                "complete": complete,
            }, fh, ensure_ascii=False, indent=2)
        print(f"報告 {args.report}")

    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
