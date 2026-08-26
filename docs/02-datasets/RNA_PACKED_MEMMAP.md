# RNA: building `training_ready_bin`

The four RNA GPT-2 configs read their split from a flat `uint16` memmap, not from
the Arrow dataset. **The memmap has to be built before any of them will start.**
`RNABinDataset` refuses to open a directory that is missing, truncated, or of the
wrong dtype rather than training on whatever it finds, so a missing build fails
loudly at startup instead of silently.

## Why

Fetching the 16-row micro-batch one rank consumes per gradient step costs 175 ms
out of the Arrow-backed HuggingFace dataset — 11 ms for an 8 KB row, which is
read latency, not bandwidth. That left 85 % of every iteration waiting on data at
~2 % MFU: 8.20 s/iter, of which 7.01 s was the fetch (job 19223, measured with a
per-section breakdown inside a real run, single job, 99 iterations).

Repacking the same rows into a flat memmap brought it to **1.36 s/iter, a factor
of 6**, which is what made the 40,320-iteration production runs fit inside the
4-day walltime at all. Every other modality already reads a flat array.

The tokens do not change. Row *i* of the `.bin` is row *i* of the split, in
order, so the same index draws the same block and training is unaffected —
verified at full scale over 15,256 rows (5,000 random per split plus both ends
and both sides of every packing-chunk boundary, job 20866, zero mismatches).
RNA's vocabulary is 25,426 entries so `uint16` is lossless, and the three splits
shrink from 329 GB to 83 GB.

## Build

```bash
export LEARNING_SOURCE_DIR=/path/to/learning_source_20260803_rna_train
SRC=/path/to/learning_source_20260723_b200prep/rna/training_ready_hf_dataset
OUT=$LEARNING_SOURCE_DIR/rna/training_ready_bin

for s in train valid test; do
    python scripts/rna_pack_to_bin.py --dataset "$SRC" --split "$s" --out "$OUT"
done
```

CPU-bound, sequential reads, no decision points — safe to leave unattended. The
production build (job 19238) took **3 h 38 min** for all three splits at ~3,300
rows/s. Add `--limit N` to pack only the first N rows for a trial — **give a trial
its own `--out`**, not the production directory.

The writer refuses to replace a build that is already at `--out`; `--overwrite`
forces it, and there is no undo. Take that refusal seriously, because nothing
after it will catch the mistake: a re-run rewrites `<split>.json` along with
`<split>.bin`, so rows, block, dtype and `max_token_id` still agree and
`RNABinDataset` opens the replacement without a word. The guards it applies on
open catch a build that is missing or truncated, not one that was quietly
rebuilt from something else.

Output, one pair per split:

```
<out>/train.bin    uint16, rows x block, C order
<out>/train.json   {"rows", "block", "dtype", "max_token_id", "source", "bytes"}
```

Expected for the 2026-07 corpus:

| split | rows | size |
|---|---|---|
| train | 34,407,040 | 70.47 GB |
| valid | 4,300,848 | 8.81 GB |
| test | 4,300,176 | 8.81 GB |

`max_token_id` is 25,406 in all three — below the 25,426-entry Geneformer
vocabulary the corpus is tokenized in.

## Wiring it up

The configs point at the directory with `rna_bin_dir`:

```python
rna_bin_dir = RNA_DATASET_DIR + "/training_ready_bin"
```

Opt-in and explicit rather than auto-detected, so the log and the config together
say which storage a run read. Comment the line out to fall back to the Arrow
path — useful for checking that the two agree, which is how the bit-identity of
the assembled batches was confirmed.

## Verifying a build

```bash
python - <<'PY'
import json, numpy as np
from datasets import load_from_disk
meta = json.load(open("<out>/train.json"))
arr = np.memmap("<out>/train.bin", dtype=np.uint16, mode="r",
                shape=(meta["rows"], meta["block"]))
hf = load_from_disk("<hf_dataset>")["train"]
ix = np.random.RandomState(0).randint(0, meta["rows"], 500).tolist()
a = np.asarray(hf[ix]["input_ids"], dtype=np.int64)
b = np.asarray(arr[ix], dtype=np.int64)
print("match:", np.array_equal(a, b))
PY
```

Sample the ends and the 20,000-row chunk boundaries too — a fencepost error in
the writer lands exactly there and a uniform random draw will miss it.
