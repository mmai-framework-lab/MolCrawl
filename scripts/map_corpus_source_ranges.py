"""Attribute rows of the combined corpus back to the source parquet they came from.

combine_all.py concatenates eight sources in a fixed order and then drops duplicate
SMILES keeping the first occurrence, so the surviving rows stay in source order and
each source occupies a contiguous range. Those ranges matter because packing reads the
parquet in row order: at 1,558 steps the run does not average over the whole corpus, so
a loss curve can be tracking which source is being read rather than anything about the
model.

Attribution is by sampling: a set of evenly spaced rows is drawn from the combined
corpus, then each source is streamed once and the sample marked. Each sampled row is
credited to the FIRST source that contains it, mirroring the keep-first dedup. That
costs one pass per source instead of holding eight SMILES sets in memory at once.

The ZINC20 member list is not reproducible -- combine_all.py samples it without a
random_state -- so ranges are derived from the built parquet rather than recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

# combine_all.py's df_list order. Attribution walks this list and stops at the first
# source containing a given SMILES, which is what keep-first dedup does.
SOURCE_ORDER = [
    "zinc_qm9", "opv", "pubchemqc_2017", "pubchemqc_2020",
    "zinc20", "reddb", "pc9", "chembl",
]


def sample_rows(parquet: Path, column: str, n_samples: int):
    """Return (row_indices, smiles) for evenly spaced rows of the combined corpus."""
    pf = pq.ParquetFile(parquet)
    total = pf.metadata.num_rows
    step = max(1, total // n_samples)
    wanted = set(range(0, total, step))
    idx, smis, pos = [], [], 0
    for batch in pf.iter_batches(batch_size=65536, columns=[column]):
        vals = batch.column(0).to_pylist()
        for j, smi in enumerate(vals):
            if pos + j in wanted and smi:
                idx.append(pos + j)
                smis.append(smi)
        pos += len(vals)
    return idx, smis, total


def mark_present(source_parquet: Path, column: str, targets: set[str]) -> set[str]:
    """Return the subset of `targets` that occurs in this source."""
    found = set()
    pf = pq.ParquetFile(source_parquet)
    for batch in pf.iter_batches(batch_size=65536, columns=[column]):
        for smi in batch.column(0).to_pylist():
            if smi in targets:
                found.add(smi)
                if len(found) == len(targets):
                    return found
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True, help="the combined corpus")
    ap.add_argument("--column", default="smiles")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                    help="repeatable; NAME must appear in the known source order")
    ap.add_argument("--source-column", default="smiles")
    ap.add_argument("--output")
    args = ap.parse_args()

    sources = {}
    for spec in args.source:
        name, _, path = spec.partition("=")
        if name not in SOURCE_ORDER:
            print(f"WARNING: {name!r} is not in the known order {SOURCE_ORDER}")
        sources[name] = Path(path)

    idx, smis, total = sample_rows(Path(args.parquet), args.column, args.samples)
    print(f"corpus {total:,} rows; sampled {len(idx):,}", flush=True)

    unresolved = set(smis)
    owner: dict[str, str] = {}
    for name in SOURCE_ORDER:
        path = sources.get(name)
        if path is None:
            print(f"  {name:16s} SKIP (path not given)")
            continue
        if not path.exists():
            print(f"  {name:16s} SKIP ({path} missing)")
            continue
        found = mark_present(path, args.source_column, unresolved)
        for smi in found:
            owner[smi] = name
        unresolved -= found
        print(f"  {name:16s} claimed {len(found):6,}   unresolved {len(unresolved):6,}", flush=True)

    # Walk the sample in row order and emit contiguous runs of the same owner.
    runs = []
    cur = None
    for row, smi in zip(idx, smis):
        who = owner.get(smi, "unattributed")
        if cur is None or cur["source"] != who:
            cur = {"source": who, "first_row": row, "last_row": row, "samples": 1}
            runs.append(cur)
        else:
            cur["last_row"] = row
            cur["samples"] += 1

    print()
    print(f"{'source':16s} {'first row':>12s} {'last row':>12s} {'rows (approx)':>14s} {'share':>7s}")
    for r in runs:
        span = r["last_row"] - r["first_row"]
        print(f"{r['source']:16s} {r['first_row']:12,} {r['last_row']:12,} {span:14,} {span/total:7.2%}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(
            {"parquet": str(args.parquet), "total_rows": total,
             "samples": len(idx), "runs": runs}, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
