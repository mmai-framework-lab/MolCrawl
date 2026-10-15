#!/usr/bin/env python3
"""Is the downstream test set inside the pretraining corpus?

The retrieval evaluation reads 1,000 held-out (molecule, description) pairs from
Mol-Instructions, and Mol-Instructions sits under the same learning_source directory the
pretraining corpus was built from. If those pairs were also pretrained on, a retrieval
score measures what the model memorised, and so does every comparison built on it.

Measured the way the train/valid overlap was (2026-09-25): each test text is tokenised
with the model's own tokenizer and its windows are looked up among the training split's,
hashed and binary-searched. Three readings, in rising order of how much they matter:

  every window the text's windows all occur in training, so it can be laid down from the
               corpus a window at a time -- the strongest sign short of a row match.
  coverage     the fraction of its windows that occur anywhere in training. A chemistry
               description scores well above zero without being copied, so read the
               distribution, not the mean.

    python scripts/molnl_downstream_contamination.py --dataset <packed train> \
        --pairs-root <Mol-Instructions dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molnl_train_valid_overlap import window_hashes  # noqa: E402  one hashing, one place


def test_texts(root, subset):
    from datasets import load_from_disk

    data = load_from_disk(os.path.join(root, subset))
    rows = []
    for item in data:
        meta = item["metadata"]
        if not isinstance(meta, dict):
            meta = json.loads(str(meta).replace("'", '"'))
        if str(meta.get("split", "")).lower() == "test":
            rows.append((item["input"], item["output"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="the packed corpus the models trained on")
    ap.add_argument("--pairs-root", required=True)
    ap.add_argument("--subset", default="molecular_description_generation")
    ap.add_argument("--split", default="train", help="the pretraining split to search")
    ap.add_argument("--column", default="input_ids")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out")
    a = ap.parse_args()

    from datasets import load_from_disk

    from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

    tokenizer = MoleculeNatLangTokenizer()
    pairs = test_texts(a.pairs_root, a.subset)
    print(f"  test pairs: {len(pairs):,}")

    train = load_from_disk(a.dataset)[a.split]
    chunks, rows_seen = [], 0
    for row in train:
        tokens = np.asarray(row[a.column], dtype=np.uint64)
        chunks.append(window_hashes(tokens, a.width))
        rows_seen += 1
    haystack = np.unique(np.concatenate(chunks))
    del chunks
    print(f"  pretraining rows {rows_seen:,}, distinct {a.width}-token windows "
          f"{haystack.size:,}")

    out = {"subset": a.subset, "pairs": len(pairs), "window_tokens": a.width,
           "pretraining_rows": rows_seen,
           "distinct_windows": int(haystack.size), "fields": {}}
    for field, index in (("molecule", 0), ("description", 1)):
        exact = 0
        coverage = []
        for pair in pairs:
            ids = tokenizer.tokenizer(pair[index])["input_ids"]
            tokens = np.asarray(ids, dtype=np.uint64)
            if tokens.size >= a.width:
                windows = window_hashes(tokens, a.width)
                idx = np.searchsorted(haystack, windows)
                idx[idx >= haystack.size] = 0
                share = float((haystack[idx] == windows).mean())
                coverage.append(share)
                if share == 1.0:
                    exact += 1
        cov = np.asarray(coverage) if coverage else np.zeros(1)
        out["fields"][field] = {
            "scored": len(coverage),
            "every_window_in_training": exact,
            "coverage_mean": round(float(cov.mean()), 4),
            "coverage_median": round(float(np.median(cov)), 4),
            "coverage_p90": round(float(np.percentile(cov, 90)), 4),
            "rows_above_0.9": int((cov > 0.9).sum()),
            "rows_above_0.99": int((cov > 0.99).sum()),
        }
        print(f"  {field}: 全窓一致 {exact} / {len(coverage)}, 被覆 平均 "
              f"{cov.mean():.4f} 中央 {np.median(cov):.4f}, 0.9 超 {(cov > 0.9).sum()}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
