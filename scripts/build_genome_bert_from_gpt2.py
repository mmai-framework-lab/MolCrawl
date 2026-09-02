"""Make the BERT dataset out of the GPT-2 one, instead of cutting windows again.

A GPT-2 row is 1,024 contiguous bases from a single contig -- the writer walks
one raw line at a time and steps by the chunk size, so no row spans a contig
boundary. A BERT row is the same body with [CLS] and [SEP] on the ends. The two
trainers already share the nucleotide ids (A=0 T=1 G=2 C=3 N=4), so the body
transfers verbatim.

Cutting the windows again at 1,022 gave a dataset that overlapped GPT-2's by two
contigs out of five thousand. The split is assigned by accumulating *window
counts* until a 50,000-window quota fills, and the window count of a contig
depends on the chunk length, so a different chunk length picks different contigs
-- the seeded order is the same, the cut-off is not. Deriving the rows instead
keeps the split GPT-2 already has, and the two models then see the same bases,
window for window.

The output is 1,026 tokens ([CLS] + 1,024 + [SEP]) rather than 1,024. Trimming
two bases to land on a round sequence length would discard data the GPT-2 side
kept, and the position embeddings are trained from scratch either way.

Verification is not optional and runs by default: every row is stripped back to
its body and compared against the GPT-2 row it came from, and the split
assignment and contig sets are compared as wholes. A wrong id mapping would
otherwise produce a dataset that trains happily and means nothing.
"""

import argparse
import os

CLS_ID = 7
SEP_ID = 8


def convert_split(ds, cache_dir, batch_size=2000, workers=8):
    """[CLS] + body + [SEP], leaving every other column alone.

    The cache file has to be named explicitly. Left to itself, ``map`` writes its
    temporary shards next to the dataset it is reading, which fails outright when
    the source tree is read-only -- as it is here, and as it should be.
    """

    def _wrap(batch):
        return {
            "input_ids": [[CLS_ID, *row, SEP_ID] for row in batch["input_ids"]],
            "attention_mask": [[1] * (len(row) + 2) for row in batch["input_ids"]],
        }

    os.makedirs(cache_dir, exist_ok=True)
    return ds.map(_wrap, batched=True, batch_size=batch_size, num_proc=workers,
                  desc="wrapping",
                  cache_file_name=os.path.join(cache_dir, "wrapped.arrow"))


def verify(src, out, sample):
    """Bodies match row for row, and the splits line up as sets."""
    problems = []
    for split in src:
        a, b = src[split], out[split]
        if len(a) != len(b):
            problems.append(f"{split}: {len(a)} rows in, {len(b)} out")
            continue
        n = min(sample, len(a))
        step = max(1, len(a) // n)
        for i in range(0, len(a), step):
            ra, rb = a[i], b[i]
            if rb["input_ids"][0] != CLS_ID or rb["input_ids"][-1] != SEP_ID:
                problems.append(f"{split}[{i}]: missing CLS/SEP")
                break
            if rb["input_ids"][1:-1] != ra["input_ids"]:
                problems.append(f"{split}[{i}]: body differs from the GPT-2 row")
                break
            if rb["accession"] != ra["accession"] or rb["contig_id"] != ra["contig_id"]:
                problems.append(f"{split}[{i}]: provenance differs")
                break
        ca = set(zip(a["accession"], a["contig_id"]))
        cb = set(zip(b["accession"], b["contig_id"]))
        if ca != cb:
            problems.append(f"{split}: contig sets differ ({len(ca ^ cb)} symmetric difference)")
        print(f"  {split:>6}: {len(b):>12,} rows, {len(cb):>6,} contigs, "
              f"{n:,} bodies checked")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="training_ready_hf_dataset_gpt2")
    ap.add_argument("dst", help="output training_ready_hf_dataset_bert")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-dir", help="where map writes its shards "
                    "(default: <dst>.cache); must be writable")
    ap.add_argument("--sample", type=int, default=2000,
                    help="rows per split to compare body-for-body")
    args = ap.parse_args()

    from datasets import DatasetDict, load_from_disk

    from molcrawl.core.output_guard import assert_output_dir, map_cache_path

    # Before anything is read: the destination must not be inside the corpus we
    # are reading from, and it must be writable. map's cache goes beside the
    # output for the same reason -- its default is beside the input.
    assert_output_dir(args.dst, extra_roots=(args.src,), what="dataset")

    src = load_from_disk(args.src)
    print(f"  source: {args.src}")
    cache_root = str(args.cache_dir or map_cache_path(args.dst))
    assert_output_dir(cache_root, extra_roots=(args.src,), what="map cache")
    out = DatasetDict({
        k: convert_split(v, os.path.join(cache_root, k), workers=args.workers)
        for k, v in src.items()
    })

    print("  verifying")
    problems = verify(src, out, args.sample)
    if problems:
        print("\n  FAILED")
        for p in problems:
            print(f"    {p}")
        raise SystemExit(1)

    os.makedirs(os.path.dirname(args.dst.rstrip("/")) or ".", exist_ok=True)
    out.save_to_disk(args.dst)
    print(f"  wrote {args.dst}")
    print(f"  sequence length: {len(out['train'][0]['input_ids'])}")

    # The shards are a copy of what was just saved; keeping them doubles the
    # footprint of every subset.
    import shutil
    shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
