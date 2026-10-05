"""Assemble the per-file ZINC-22 shards into one DatasetDict for training.

The shards are already packed into fixed blocks, so this step only has to decide
which block goes to which split and write them out in the layout the two trainers
load. It is separate from packing because it is the one step that has to see every
shard, and because the split has to be reproducible: blocks are assigned by hashing
the shard name and the block index, so adding a shard later does not move the blocks
already assigned, and re-running produces the same split without holding the corpus
in memory.

The proportions and the field layout follow organix13's training_ready_hf_dataset:
80/10/10 over blocks, one int32 `input_ids` column.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "valid", "test")


def split_of(shard_name: str, index: int, weights) -> str:
    """Which split this block belongs to -- a function of its identity, not of order."""
    digest = hashlib.blake2b(f"{shard_name}:{index}".encode(), digest_size=8).digest()
    point = int.from_bytes(digest, "big") / float(1 << 64)
    acc = 0.0
    for name, w in zip(SPLITS, weights):
        acc += w
        if point < acc:
            return name
    return SPLITS[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="directory of parquet shards")
    ap.add_argument("--out", required=True, help="DatasetDict directory to write")
    ap.add_argument("--valid-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--rows-per-group", type=int, default=16_384)
    args = ap.parse_args(argv)

    weights = (1.0 - args.valid_fraction - args.test_fraction,
               args.valid_fraction, args.test_fraction)
    if min(weights) <= 0:
        print("valid と test の割合の合計が 1 以上です", flush=True)
        return 2

    shards = sorted(glob.glob(os.path.join(args.shards, "*.parquet")))
    if not shards:
        print(f"shard が無い: {args.shards}", flush=True)
        return 1
    print(f"shard {len(shards):,} 本 / 割合 {weights}", flush=True)

    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])
    writers, counts = {}, dict.fromkeys(SPLITS, 0)
    for name in SPLITS:
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        writers[name] = pq.ParquetWriter(os.path.join(d, "data.parquet"), schema,
                                         compression="zstd")

    try:
        for n, path in enumerate(shards, 1):
            stem = os.path.basename(path)
            table = pq.read_table(path, columns=["input_ids"])
            # One pass over the shard, collecting each split's row indices, so the
            # blocks are taken as a batch rather than row by row.
            buckets = {name: [] for name in SPLITS}
            for i in range(table.num_rows):
                buckets[split_of(stem, i, weights)].append(i)
            for name, idx in buckets.items():
                if not idx:
                    continue
                part = table.take(idx)
                writers[name].write_table(part.cast(schema),
                                          row_group_size=args.rows_per_group)
                counts[name] += len(idx)
            if n % 50 == 0 or n == len(shards):
                total = sum(counts.values())
                print(f"  {n:,}/{len(shards):,} shard  ブロック {total:,}  "
                      + " ".join(f"{k} {v:,}" for k, v in counts.items()), flush=True)
    finally:
        for w in writers.values():
            w.close()

    total = sum(counts.values())
    summary = {
        "shards": len(shards),
        "blocks": counts,
        "blocks_total": total,
        "fractions": {k: round(v / total, 5) for k, v in counts.items()} if total else {},
        "block_size_tokens": 1024,
        "tokens_total": total * 1024,
    }
    with open(os.path.join(args.out, "assemble_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\nブロック {total:,} = {total * 1024:,} トークン")
    for k, v in counts.items():
        print(f"  {k:5s} {v:,} ({v / total:.4%})" if total else f"  {k}: 0")
    print(f"出力 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
