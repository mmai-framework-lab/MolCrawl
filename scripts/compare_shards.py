"""Are two packed shards the same corpus, token for token?

Written to check that packing a tranche with a worker pool gives the same blocks as
packing it with one process. Matching block counts do not establish that: two
different cuts of the same token stream can agree on how many blocks they produce
and still disagree on what is in them, so the comparison has to run over the tokens.

Hashes each shard's blocks in order and prints both digests, so a mismatch is visible
without holding either shard in memory twice.
"""

from __future__ import annotations

import argparse
import hashlib

import pyarrow.parquet as pq


def digest(path):
    """Row count and a digest of every token, in block order."""
    table = pq.read_table(path, columns=["input_ids"])
    h = hashlib.blake2b(digest_size=16)
    for row in table.column("input_ids").to_pylist():
        h.update(b"".join(int(x).to_bytes(2, "little") for x in row))
    return table.num_rows, h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("left")
    ap.add_argument("right")
    args = ap.parse_args(argv)

    n_left, h_left = digest(args.left)
    n_right, h_right = digest(args.right)
    print(f"{n_left:,} 行  {h_left}  {args.left}")
    print(f"{n_right:,} 行  {h_right}  {args.right}")
    same = (n_left, h_left) == (n_right, h_right)
    print("一致" if same else "不一致")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
