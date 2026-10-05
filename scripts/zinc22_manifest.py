"""Write one manifest describing the ZINC-22 corpus that was actually built.

The acquisition record, the per-file packing counts and the assemble summary each
describe one stage, and none of them alone says what the corpus is. This joins them
so the corpus can be cited by its own numbers rather than by the numbers that were
projected before it was built, and so a later run can be compared against it.

Where a stage is incomplete the manifest says so rather than filling the gap: a
missing shard is the difference between what was fetched and what was packed, which
is worth seeing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone


def read_records(paths):
    """Collapse several fetch records to the last word on each file."""
    status = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                status[(row["tranche"], row["file"])] = row
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--record", nargs="+", required=True)
    ap.add_argument("--counts", required=True, help="directory of per-file counts JSON")
    ap.add_argument("--assemble-summary", help="assemble_summary.json, when assembled")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(args.inventory, encoding="utf-8") as fh:
        listed = list(csv.DictReader(fh, delimiter="\t"))
    listed_bytes = sum(int(r["size_bytes"]) for r in listed)

    status = read_records(args.record)
    held = [k for k, v in status.items() if v["status"] in ("taken", "skipped")]
    failed = [k for k, v in status.items() if v["status"] == "failed"]

    counts = []
    for path in sorted(glob.glob(os.path.join(args.counts, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            counts.append(json.load(fh))

    molecules = sum(c["molecules"] for c in counts)
    tokens = sum(c["tokens"] for c in counts)
    blocks = sum(c["blocks"] for c in counts)
    unknown = sum(c["molecules_with_unknown_token"] for c in counts)
    charged = sum(c["molecules_with_charge_sign"] for c in counts)
    fragments = sum(c["molecules_multi_fragment"] for c in counts)
    dropped = sum(c["dropped_tail_tokens"] for c in counts)

    assembled = None
    if args.assemble_summary and os.path.exists(args.assemble_summary):
        with open(args.assemble_summary, encoding="utf-8") as fh:
            assembled = json.load(fh)

    manifest = {
        "written": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": "ZINC-22 2D tranches, files.docking.org",
        "inventory": {"files": len(listed), "bytes": listed_bytes},
        "acquisition": {
            "files_held": len(held),
            "files_failed": len(failed),
            "complete": len(held) == len(listed),
        },
        "packing": {
            "shards": len(counts),
            "files_not_packed": len(held) - len(counts),
            "molecules": molecules,
            "tokens": tokens,
            "blocks": blocks,
            "block_size_tokens": counts[0]["block_size"] if counts else None,
            "tokens_per_molecule": round(tokens / molecules, 3) if molecules else None,
            "molecules_with_unknown_token": unknown,
            "molecules_with_charge_sign": charged,
            "molecules_multi_fragment": fragments,
            "dropped_tail_tokens": dropped,
            "ions_removed": False,
        },
        "splits": assembled["blocks"] if assembled else None,
        "split_fractions": assembled["fractions"] if assembled else None,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"一覧      {len(listed):,} ファイル / {listed_bytes / 1024**4:.2f} TiB")
    print(f"取得      {len(held):,} 保持 / {len(failed):,} 失敗"
          f"{'  （完全）' if manifest['acquisition']['complete'] else '  （不完全）'}")
    print(f"packing   shard {len(counts):,} / 未加工 {len(held) - len(counts):,}")
    if molecules:
        print(f"          分子 {molecules:,} / トークン {tokens:,} "
              f"/ 1 分子あたり {tokens / molecules:.2f}")
        print(f"          ブロック {blocks:,} / 未知トークンを含む分子 {unknown:,} "
              f"({unknown / molecules:.4%})")
        print(f"          電荷記号あり {charged:,} ({charged / molecules:.3%}) "
              f"/ 複数断片 {fragments:,} ({fragments / molecules:.3%})")
    if assembled:
        print("          " + " ".join(f"{k} {v:,}" for k, v in assembled["blocks"].items()))
    print(f"台帳 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
