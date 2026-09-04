"""Check a derived BERT dataset against the GPT-2 one it came from.

The conversion is a wrap, not a transform: [CLS] and [SEP] around a body that
should be the GPT-2 row unchanged. Two things can still go wrong, and neither
announces itself.

Row order. ``map`` runs across processes and writes shards; if the shards were
reassembled out of order the bodies would still all be present, the contig sets
would still match, and every row would be paired with the wrong provenance. This
walks positions, not contents, so a permutation shows up.

Split inheritance. Taking GPT-2's split is the whole point -- it is what makes
the two models comparable -- so the contig sets are compared as sets, per split,
rather than trusted because the counts agree.

The dtype and declared length are checked too. They do not affect what a model
learns, but int64 doubles the file and a missing length hides the window size.
"""

import argparse
import random

# valid and test are compared in full. Every number this campaign reports is
# computed on those rows, so a single row of the wrong provenance changes a
# reported value -- and at 50,000 rows each the exhaustive check is cheap.
EXHAUSTIVE = ("valid", "test")

# Rows are pulled in batches so an exhaustive pass does not hold a whole split
# of Python lists at once.
READ_CHUNK = 5000


def _positions(split, n, args):
    """Which row positions to compare, and a phrase describing the choice.

    A stride sample on its own steps over a local reordering: it takes one row
    every `step`, so a run of rows swapped among themselves is very likely to
    fall entirely between two sampled positions. That is the failure this script
    exists to catch, because datasets.map runs per-process over contiguous shards
    and concatenates them, so the way it can go wrong is precisely local.

    Contiguous blocks close that gap. The stride says the split as a whole is
    aligned; the blocks say the rows inside it are in order.
    """
    if split in EXHAUSTIVE or n <= args.sample:
        return list(range(n)), f"all {n:,} rows"

    step = max(1, n // args.sample)
    idx = set(range(0, n, step))
    strided = len(idx)

    size = min(args.block_size, n)
    rng = random.Random(args.seed)
    for _ in range(args.blocks):
        start = rng.randrange(0, n - size + 1)
        idx.update(range(start, start + size))

    return sorted(idx), (f"{strided:,} at stride {step:,}, plus {args.blocks}"
                         f" contiguous blocks of {size:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpt2")
    ap.add_argument("bert")
    ap.add_argument("--sample", type=int, default=100000,
                    help="stride positions to compare in a sampled split")
    ap.add_argument("--blocks", type=int, default=100,
                    help="contiguous runs to compare on top of the stride")
    ap.add_argument("--block-size", type=int, default=1000,
                    help="rows per contiguous run")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for where the contiguous runs start")
    args = ap.parse_args()

    from datasets import load_from_disk

    g = load_from_disk(args.gpt2)
    b = load_from_disk(args.bert)
    problems = []

    for split in g:
        gs, bs = g[split], b[split]
        print(f"\n  ## {split}")

        if len(gs) != len(bs):
            problems.append(f"{split}: {len(gs)} rows in, {len(bs)} out")
            continue
        print(f"    rows: {len(bs):,}")

        f = bs.features["input_ids"]
        dtype = f.feature.dtype
        length = getattr(f, "length", None)
        print(f"    input_ids: {dtype}, length {length}")
        if dtype != "int32":
            problems.append(f"{split}: input_ids is {dtype}, expected int32")
        gl = getattr(gs.features["input_ids"], "length", None)
        if gl and length != gl + 2:
            problems.append(f"{split}: length {length}, expected {gl} + 2")

        # Positions, not contents: a reordered shard passes a contents check.
        idx, how = _positions(split, len(gs), args)
        bad_order = bad_body = bad_ends = 0
        for lo in range(0, len(idx), READ_CHUNK):
            part = idx[lo:lo + READ_CHUNK]
            gb, bb = gs[part], bs[part]
            for i in range(len(part)):
                if (gb["accession"][i] != bb["accession"][i]
                        or gb["contig_id"][i] != bb["contig_id"][i]):
                    bad_order += 1
                row = bb["input_ids"][i]
                if row[0] != 7 or row[-1] != 8:
                    bad_ends += 1
                elif list(row[1:-1]) != list(gb["input_ids"][i]):
                    bad_body += 1
        print(f"    checked {len(idx):,} positions ({how}): "
              f"provenance mismatches {bad_order}, "
              f"CLS/SEP {bad_ends}, bodies {bad_body}")
        if bad_order:
            problems.append(f"{split}: {bad_order} rows sit at a different position "
                            f"than in the source -- shards reassembled out of order")
        if bad_ends:
            problems.append(f"{split}: {bad_ends} rows missing CLS/SEP")
        if bad_body:
            problems.append(f"{split}: {bad_body} bodies differ from the GPT-2 row")

        cg = set(zip(gs["accession"], gs["contig_id"]))
        cb = set(zip(bs["accession"], bs["contig_id"]))
        print(f"    contigs: {len(cb):,}, identical to source: {cg == cb}")
        if cg != cb:
            problems.append(f"{split}: contig sets differ by {len(cg ^ cb)}")

    print()
    if problems:
        print("  FAILED")
        for p in problems:
            print(f"    {p}")
        raise SystemExit(1)
    print("  all checks passed")


if __name__ == "__main__":
    main()
