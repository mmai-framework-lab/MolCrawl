"""Fit n-gram baselines on the packed train split and score them on valid.

GPT-2's loss says how many nats a token costs, but not how many it could have cost.
Without a floor, "0.5921" is a number with nothing to compare it to, and the spread
between ladder sizes cannot be judged against the room that was left.

The baselines here are the CLM counterparts of the MLM look-up baselines: they use
LEFT context only, because that is all a causal model sees. The MLM figures already
on record (compounds unigram 2.483, degenerate 2.234) are not comparable -- they score
masked positions with context on both sides.

Counts come from train, cross-entropy is measured on valid, so a baseline cannot win
by memorising the split it is scored on. Add-k smoothing keeps unseen contexts finite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def flat_tokens(dataset_dir: Path, split: str, max_sequences: int | None) -> np.ndarray:
    ds = load_from_disk(str(dataset_dir / split))
    n = len(ds) if max_sequences is None else min(len(ds), max_sequences)
    print(f"  {split}: {n:,} sequences", flush=True)
    chunks = []
    step = 20000
    for start in range(0, n, step):
        part = ds[start:min(start + step, n)]["input_ids"]
        chunks.append(np.asarray(part, dtype=np.int32).ravel())
    return np.concatenate(chunks)


def counts(flat: np.ndarray, vocab: int, order: int) -> np.ndarray:
    """Dense count table over `order`-grams, flattened. order 1 -> V, order 2 -> V*V."""
    if order == 1:
        return np.bincount(flat, minlength=vocab).astype(np.float64)
    idx = flat[:-1].astype(np.int64) * vocab + flat[1:].astype(np.int64)
    return np.bincount(idx, minlength=vocab * vocab).astype(np.float64)


def cross_entropy_unigram(train_c: np.ndarray, valid: np.ndarray, k: float) -> float:
    p = (train_c + k) / (train_c.sum() + k * train_c.size)
    return float(-np.log(p[valid]).mean())


def cross_entropy_bigram(train_c2: np.ndarray, vocab: int, valid: np.ndarray, k: float) -> float:
    ctx_total = train_c2.reshape(vocab, vocab).sum(axis=1)
    total = 0.0
    n = 0
    step = 5_000_000
    for start in range(0, valid.size - 1, step):
        prev = valid[start:start + step].astype(np.int64)
        nxt = valid[start + 1:start + 1 + prev.size].astype(np.int64)
        prev = prev[: nxt.size]
        num = train_c2[prev * vocab + nxt] + k
        den = ctx_total[prev] + k * vocab
        total += float(-np.log(num / den).sum())
        n += nxt.size
    return total / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True, help="packed HF dataset root (train/ valid/)")
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--max-train-sequences", type=int, default=None)
    ap.add_argument("--max-valid-sequences", type=int, default=None)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--output")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    print("loading token streams", flush=True)
    train = flat_tokens(root, "train", args.max_train_sequences)
    valid = flat_tokens(root, "valid", args.max_valid_sequences)
    print(f"  train {train.size:,} tokens / valid {valid.size:,} tokens", flush=True)

    hi = int(max(train.max(), valid.max()))
    if hi >= args.vocab:
        print(f"WARNING: token id {hi} >= --vocab {args.vocab}; widening")
    vocab = max(args.vocab, hi + 1)

    print("counting", flush=True)
    c1 = counts(train, vocab, 1)
    c2 = counts(train, vocab, 2)

    # The token marginal, applied regardless of context. Predicting it is the
    # degenerate solution for a causal model -- the same role the MLM degenerate
    # line plays, but computed on next-token prediction.
    uni = cross_entropy_unigram(c1, valid, args.smoothing)
    bi = cross_entropy_bigram(c2, vocab, valid, args.smoothing)
    observed_vocab = int((c1 > 0).sum())
    uniform = float(np.log(observed_vocab))

    rows = [
        ("uniform over observed vocab", uniform),
        ("unigram (degenerate solution)", uni),
        ("bigram, left context only", bi),
    ]
    print()
    for name, v in rows:
        print(f"  {name:34s} {v:8.4f} nat/token")
    print()
    print(f"  observed vocab: {observed_vocab} of {vocab}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "dataset_dir": str(root), "vocab": vocab, "observed_vocab": observed_vocab,
            "train_tokens": int(train.size), "valid_tokens": int(valid.size),
            "smoothing": args.smoothing,
            "baselines": {name: value for name, value in rows},
        }, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
