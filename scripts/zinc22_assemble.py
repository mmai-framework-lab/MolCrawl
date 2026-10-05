"""Assign each packed ZINC-22 block to a split, one shard at a time.

The shards are already packed into fixed blocks, so this step only has to decide
which block goes to which split. Two properties are wanted and both come from
deciding it per shard: the assignment has to be reproducible, and adding a shard
later must not move the blocks already assigned.

So the seed is derived from the shard's name and nothing else. A shard's draws are
then generated in one vectorised call rather than hashed block by block, which is
what makes the full corpus tractable: hashing each of roughly four billion blocks
ran at twenty-odd thousand a second, or days, while the draw costs almost nothing
next to reading the shard.

Each shard writes its own three files, so this runs as an array over the shards with
no two tasks touching the same output. The result is a directory per split holding
one parquet file per shard, which is what the trainers load. Nothing has to see every
shard, so there is no assembling step left to serialise on.

Proportions and field layout follow organix13's training_ready_hf_dataset: 80/10/10
over blocks, one int32 `input_ids` column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "valid", "test")


def seed_for(shard_name: str) -> int:
    """A shard's seed, fixed by its name so the assignment never depends on order."""
    return int.from_bytes(hashlib.blake2b(shard_name.encode(), digest_size=8).digest(),
                          "big") % (2 ** 63)


def assign(shard_name: str, n_rows: int, weights) -> np.ndarray:
    """Which split each of this shard's blocks belongs to, as indices into SPLITS."""
    rng = np.random.default_rng(seed_for(shard_name))
    draws = rng.random(n_rows)
    edges = np.cumsum(weights[:-1])
    return np.searchsorted(edges, draws, side="right")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True, help="one parquet shard")
    ap.add_argument("--out", required=True, help="dataset root; one directory per split")
    ap.add_argument("--counts", required=True, help="JSON of what this shard contributed")
    ap.add_argument("--valid-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--rows-per-group", type=int, default=16_384)
    args = ap.parse_args(argv)

    weights = np.array([1.0 - args.valid_fraction - args.test_fraction,
                        args.valid_fraction, args.test_fraction])
    if weights.min() <= 0:
        print("valid と test の割合の合計が 1 以上です")
        return 2

    stem = os.path.basename(args.shard)
    if os.path.exists(args.counts):
        print(f"既存 {args.counts} -- 何もしない")
        return 0

    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])
    table = pq.read_table(args.shard, columns=["input_ids"])
    which = assign(stem, table.num_rows, weights)

    counts = {}
    for i, name in enumerate(SPLITS):
        idx = np.flatnonzero(which == i)
        counts[name] = int(idx.size)
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, stem)
        if idx.size:
            pq.write_table(table.take(idx).cast(schema), out,
                           compression="zstd", row_group_size=args.rows_per_group)
        else:
            # An empty split still gets a file, so a reader never has to know which
            # shards happened to contribute nothing.
            pq.write_table(schema.empty_table(), out, compression="zstd")

    total = table.num_rows
    summary = {
        "shard": os.path.abspath(args.shard),
        "seed": seed_for(stem),
        "blocks": counts,
        "blocks_total": total,
        "fractions": {k: round(v / total, 5) for k, v in counts.items()} if total else {},
    }
    os.makedirs(os.path.dirname(args.counts) or ".", exist_ok=True)
    with open(args.counts, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"{stem}  ブロック {total:,}  "
          + "  ".join(f"{k} {v:,}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
