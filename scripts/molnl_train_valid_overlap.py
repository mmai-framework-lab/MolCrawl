#!/usr/bin/env python3
"""How much of the evaluation split also appears in the training split.

medium at 3e-4 reaches eval_loss_mask 0.0670, a 58th of the unigram floor: roughly 93%
of masked tokens predicted correctly. The evaluation split is 1,631 rows. If rows like
them are in the training data, that number is partly a measure of what the model
memorised, and so is every comparison built on it (2026-09-25 order §2.5).

Three things are measured, in rising order of how much they should worry anyone:

  exact rows        an evaluation row whose tokens are identical to a training row.
  shared prefixes   identical first 512 tokens. Same document, later divergence.
  n-gram coverage   the fraction of an evaluation row's 16-token windows that occur
                    somewhere in training. A row at 1.0 is reproducible from the
                    training data a window at a time, whether or not any row matches it
                    whole; a row of ordinary English still scores well above 0, so read
                    the distribution rather than the mean.

The windows are hashed, not stored: 64-bit rolling hashes over the training split, sorted
once, then looked up by binary search. Collisions at 2^-64 per pair do not move a
fraction reported to two decimals.

    python scripts/molnl_train_valid_overlap.py --dataset <dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

MULT = np.uint64(0x100000001B3)     # FNV's multiplier, used as a rolling mixer
SEED = np.uint64(0xCBF29CE484222325)


def _rows(split, column):
    for row in split:
        yield np.asarray(row[column], dtype=np.uint64)


def window_hashes(tokens: np.ndarray, width: int) -> np.ndarray:
    """Hash every ``width``-token window of a 1-D token array."""
    if tokens.size < width:
        return np.empty(0, dtype=np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(tokens, width)
    digest = np.full(windows.shape[0], SEED, dtype=np.uint64)
    for i in range(width):
        digest = (digest ^ windows[:, i]) * MULT
    return digest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="valid")
    ap.add_argument("--column", default="input_ids")
    ap.add_argument("--width", type=int, default=16, help="tokens per window")
    ap.add_argument("--prefix", type=int, default=512, help="tokens compared for a shared prefix")
    ap.add_argument("--out")
    a = ap.parse_args()

    from datasets import load_from_disk

    data = load_from_disk(a.dataset)
    train, evalset = data[a.train_split], data[a.eval_split]

    # Training side: whole-row hashes, prefix hashes, and every window hash.
    train_rows, train_prefixes, chunks = set(), set(), []
    n_train = 0
    for tokens in _rows(train, a.column):
        train_rows.add(int(window_hashes(tokens, tokens.size)[0]) if tokens.size else 0)
        if tokens.size >= a.prefix:
            train_prefixes.add(int(window_hashes(tokens[:a.prefix], a.prefix)[0]))
        chunks.append(window_hashes(tokens, a.width))
        n_train += 1
    train_windows = np.unique(np.concatenate(chunks)) if chunks else np.empty(0, np.uint64)
    del chunks

    exact = prefix_shared = 0
    coverage = []
    n_eval = 0
    for tokens in _rows(evalset, a.column):
        n_eval += 1
        if tokens.size and int(window_hashes(tokens, tokens.size)[0]) in train_rows:
            exact += 1
        if tokens.size >= a.prefix and \
                int(window_hashes(tokens[:a.prefix], a.prefix)[0]) in train_prefixes:
            prefix_shared += 1
        windows = window_hashes(tokens, a.width)
        if windows.size:
            idx = np.searchsorted(train_windows, windows)
            idx[idx >= train_windows.size] = 0
            coverage.append(float((train_windows[idx] == windows).mean()))

    cov = np.asarray(coverage) if coverage else np.zeros(1)
    result = {
        "dataset": os.path.abspath(a.dataset),
        "train_rows": n_train, "eval_rows": n_eval,
        "window_tokens": a.width, "prefix_tokens": a.prefix,
        "distinct_train_windows": int(train_windows.size),
        "eval_rows_identical_to_a_train_row": exact,
        "eval_rows_identical_to_a_train_row_pct": round(100 * exact / max(n_eval, 1), 3),
        "eval_rows_sharing_a_train_prefix": prefix_shared,
        "eval_rows_sharing_a_train_prefix_pct": round(100 * prefix_shared / max(n_eval, 1), 3),
        "window_coverage": {
            "mean": round(float(cov.mean()), 4),
            "median": round(float(np.median(cov)), 4),
            "p90": round(float(np.percentile(cov, 90)), 4),
            "max": round(float(cov.max()), 4),
            "rows_above_0.9": int((cov > 0.9).sum()),
            "rows_above_0.99": int((cov > 0.99).sum()),
        },
    }
    print(json.dumps(result, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
