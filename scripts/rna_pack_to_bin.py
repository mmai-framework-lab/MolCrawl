"""Pack an RNA training_ready split into a flat uint16 memmap.

RNA GPT-2 could not be trained from the Arrow-backed HuggingFace dataset in any
reasonable time. Fetching the 16-row micro-batch that one rank consumes per
gradient step costs 175 ms out of Arrow — 11 ms for an 8 KB row, i.e. read
latency, not bandwidth — and that left 85 % of every iteration waiting on data
at ~2 % MFU (measured in job 19223: 8.20 s/iter, of which 7.01 s was the fetch).
Repacking the same rows into a flat memmap brought it to 1.36 s/iter, a factor
of 6, which is what made the 40,320-iteration production runs fit inside the
4-day walltime at all.

The tokens do not change. Row i of the ``.bin`` is row i of the split, in order,
so the same index draws the same block and training is unaffected — verified at
full scale over 15,256 rows (5,000 random per split plus both ends and both
sides of every packing-chunk boundary). RNA's vocabulary is 25,426 entries, so
uint16 is lossless, and the three splits shrink from 329 GB to 83 GB.

Output layout, one pair per split, next to each other::

    <out>/train.bin   uint16, rows x block, C order
    <out>/train.json  {"rows", "block", "dtype", "max_token_id", "source", "bytes"}

Point a config at the directory with ``rna_bin_dir`` to use it; drop that line
and the loader falls back to Arrow. The four RNA GPT-2 configs ship with it set,
so **this script has to be run before any of them will start** — RNABinDataset
refuses to open a directory that is missing, truncated, or of the wrong dtype
rather than training on whatever it finds.

Usage (the full corpus takes ~3.6 h for all three splits, CPU-bound, sequential
reads — safe to leave unattended)::

    python scripts/rna_pack_to_bin.py \
        --dataset $LEARNING_SOURCE_DIR/rna/training_ready_hf_dataset \
        --split train --out $LEARNING_SOURCE_DIR/rna/training_ready_bin

    # all three, as the production build did (job 19238):
    for s in train valid test; do
        python scripts/rna_pack_to_bin.py --dataset <hf_dataset> --split $s --out <dir>
    done

``--limit N`` packs only the first N rows, for a trial build.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

UINT16_MAX = 65535


def pack(dataset_dir, split, out_dir, limit=None, chunk=20000):
    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_dir))[split]
    n = len(ds) if limit is None else min(limit, len(ds))
    block = len(ds[0]["input_ids"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{split}.bin"

    arr = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(n, block))
    t0 = time.time()
    seen_max = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = ds[start:stop]["input_ids"]          # sequential: this part is fast
        block_arr = np.asarray(rows, dtype=np.int64)
        if block_arr.shape[1] != block:
            raise ValueError(f"ragged rows at {start}: {block_arr.shape[1]} != {block}")
        hi = int(block_arr.max())
        if hi > UINT16_MAX:
            raise ValueError(f"token id {hi} does not fit in uint16")
        seen_max = max(seen_max, hi)
        arr[start:stop] = block_arr.astype(np.uint16)
        if start % (chunk * 20) == 0:
            done = stop / n
            el = time.time() - t0
            print(f"  {stop:,}/{n:,} ({done*100:5.1f}%) {el:6.1f}s "
                  f"eta {el/max(done,1e-9)-el:6.1f}s", flush=True)
    arr.flush()
    del arr

    meta = {
        "split": split,
        "rows": n,
        "block": block,
        "dtype": "uint16",
        "max_token_id": seen_max,
        "source": str(dataset_dir),
        "bytes": bin_path.stat().st_size,
    }
    (out_dir / f"{split}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {bin_path} ({meta['bytes']/1e9:.2f} GB) in {time.time()-t0:.1f}s")
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    pack(a.dataset, a.split, a.out, a.limit)
