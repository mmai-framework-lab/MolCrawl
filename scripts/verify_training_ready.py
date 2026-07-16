"""Standing gate: verify a training_ready dataset via `load_from_disk` round-trip.

Boss requirement (Phase 1-2 review, Q4 condition 2):
> `load_from_disk` による検算 (行数・seq_len・pad 除外 sum) を受け入れゲート
> として常設。 フォーマット差異が出たら必ずここで気付けるようにする。

Usage:
    python3 tmp/scripts/verify_training_ready.py <dataset_dir> [--seq-len 128] [--pad-id 0]

Exits 0 on all-PASS, non-zero on any failure.

For each split in the DatasetDict:
  1. Load via datasets.load_from_disk (validates HF sidecar format).
  2. Assert row count > 0.
  3. Sample rows to confirm seq_len is uniform.
  4. Full pass over the split summing pad-excluded token counts.

Prints a compact PASS report and a machine-readable JSON summary.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir", help="path to training_ready_hf_dataset (root of DatasetDict)")
    ap.add_argument("--seq-len", type=int, default=128,
                    help="expected fixed seq_len for every row (default 128)")
    ap.add_argument("--pad-id", type=int, default=0,
                    help="pad token id for the pad-excluded sum (default 0)")
    ap.add_argument("--json", action="store_true",
                    help="also emit the summary as a JSON block on the last line")
    args = ap.parse_args()

    from datasets import load_from_disk
    import datasets as hfds

    print(f"[verify] hf datasets version: {hfds.__version__}")
    print(f"[verify] dataset dir: {args.dataset_dir}")
    print(f"[verify] expected seq_len={args.seq_len}, pad_id={args.pad_id}")
    t0 = time.time()

    d = load_from_disk(args.dataset_dir)
    if not hasattr(d, "keys"):
        print("[FAIL] loaded object is not a DatasetDict")
        return 2

    summary = {"hf_datasets_version": hfds.__version__,
               "dataset_dir": args.dataset_dir,
               "seq_len_expected": args.seq_len,
               "pad_id": args.pad_id,
               "splits": {}}

    all_pass = True
    for sp in sorted(d.keys()):
        s = d[sp]
        n = len(s)
        if n == 0:
            print(f"[FAIL] {sp}: 0 rows")
            all_pass = False
            continue

        cols = set(s.column_names)
        if "input_ids" not in cols:
            print(f"[FAIL] {sp}: missing input_ids column (has {cols})")
            all_pass = False
            continue

        # Sample first, middle, last row for seq_len uniformity
        sample_idxs = [0, n // 2, n - 1]
        sample_lens = [len(s[i]["input_ids"]) for i in sample_idxs]
        if any(L != args.seq_len for L in sample_lens):
            print(f"[FAIL] {sp}: sample seq_len {sample_lens} != {args.seq_len}")
            all_pass = False
            continue

        # Full pass: pad-excluded token count
        pad_incl = 0
        pad_excl = 0
        batch = 10_000
        for st in range(0, n, batch):
            end = min(st + batch, n)
            a = np.asarray(s[st:end]["input_ids"], dtype=np.int64)
            pad_incl += int(a.size)
            pad_excl += int((a != args.pad_id).sum())

        split_row = {
            "rows": n,
            "seq_len": args.seq_len,
            "pad_incl_tokens": pad_incl,
            "pad_excl_tokens": pad_excl,
            "pad_excl_pct": round(100 * pad_excl / pad_incl, 3),
            "has_attention_mask": "attention_mask" in cols,
        }
        summary["splits"][sp] = split_row
        print(f"[PASS] {sp}: rows={n:,}  seq_len={args.seq_len}  "
              f"pad_incl={pad_incl:,}  pad_excl={pad_excl:,}  "
              f"({split_row['pad_excl_pct']}% real)"
              f"{'  attention_mask✓' if split_row['has_attention_mask'] else ''}")

    print(f"[verify] elapsed {time.time()-t0:.1f}s, overall: {'PASS' if all_pass else 'FAIL'}")
    if args.json:
        print("__SUMMARY_JSON__ " + json.dumps(summary))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
