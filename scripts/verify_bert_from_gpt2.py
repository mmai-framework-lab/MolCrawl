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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpt2")
    ap.add_argument("bert")
    ap.add_argument("--sample", type=int, default=20000,
                    help="positions per split to compare row for row")
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
        n = min(args.sample, len(gs))
        step = max(1, len(gs) // n)
        idx = list(range(0, len(gs), step))
        gb, bb = gs[idx], bs[idx]
        bad_order = bad_body = bad_ends = 0
        for i in range(len(idx)):
            if (gb["accession"][i] != bb["accession"][i]
                    or gb["contig_id"][i] != bb["contig_id"][i]):
                bad_order += 1
            row = bb["input_ids"][i]
            if row[0] != 7 or row[-1] != 8:
                bad_ends += 1
            elif list(row[1:-1]) != list(gb["input_ids"][i]):
                bad_body += 1
        print(f"    checked {len(idx):,} positions: "
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
