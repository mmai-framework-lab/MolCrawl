#!/usr/bin/env python3
"""Measure what a model that reads nothing scores on this corpus.

The GPT-2 figures had no reference line. The 3.8638 measured for molecule_nat_lang is the
unigram floor over masked positions, which is the reference for BERT's eval_loss_mask and
not for next-token loss: different positions, different metric (2026-09-25 order §5).

What is measured here: predict every token by the corpus's token frequencies, ignoring
all context, and take the cross-entropy on the evaluation split. Anything a trained model
scores above this it could have beaten by counting tokens once and reading nothing.

Two numbers come out, and they answer different questions.

  train-frequency floor   frequencies counted on train, scored on valid. This is the one
                          to draw: it is what a context-free model fitted on the training
                          data actually achieves on held-out data.
  valid's own entropy     frequencies counted on valid, scored on valid. No context-free
                          model can beat it on this split. The gap between the two is how
                          much the split's token distribution differs from the training
                          one.

Counts cover every token of the split. Next-token prediction scores every position after
the first of each sequence; for a model with no context the position does not matter, so
the two differ only by one token per sequence, and the run prints both counts to show it.

    python scripts/molnl_unigram_floor.py --dataset <training_ready_hf_dataset_shuffled>
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os


def count_tokens(split, column):
    """Token -> occurrences, over every row of a split."""
    counts = collections.Counter()
    rows = 0
    for row in split:
        counts.update(row[column])
        rows += 1
    return counts, rows


def cross_entropy(counts_model, counts_data, vocab_size, smoothing=1.0):
    """Mean -log p over the data, with p from the model's counts.

    Add-one smoothing over the whole vocabulary, so a token the model never saw costs a
    large number rather than infinity. With 326 M training tokens it moves the result far
    below the digits reported here; it is there so one unseen token cannot make the
    answer meaningless.
    """
    total = sum(counts_model.values()) + smoothing * vocab_size
    n = sum(counts_data.values())
    nll = 0.0
    for token, times in counts_data.items():
        p = (counts_model.get(token, 0) + smoothing) / total
        nll -= times * math.log(p)
    return nll / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="a load_from_disk directory with splits")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="valid")
    ap.add_argument("--column", default="input_ids")
    ap.add_argument("--vocab-size", type=int, default=50257, help="GPT-2's vocabulary")
    ap.add_argument("--out", help="write the numbers here as json")
    a = ap.parse_args()

    from datasets import load_from_disk

    data = load_from_disk(a.dataset)
    train_counts, train_rows = count_tokens(data[a.train_split], a.column)
    eval_counts, eval_rows = count_tokens(data[a.eval_split], a.column)

    train_floor = cross_entropy(train_counts, eval_counts, a.vocab_size)
    eval_entropy = cross_entropy(eval_counts, eval_counts, a.vocab_size)
    eval_tokens = sum(eval_counts.values())

    result = {
        "dataset": os.path.abspath(a.dataset),
        "train_split": a.train_split, "eval_split": a.eval_split,
        "train_rows": train_rows, "eval_rows": eval_rows,
        "train_tokens": sum(train_counts.values()), "eval_tokens": eval_tokens,
        "eval_types": len(eval_counts), "train_types": len(train_counts),
        "vocab_size": a.vocab_size,
        "unigram_floor_train_frequencies": round(train_floor, 4),
        "entropy_of_eval_split": round(eval_entropy, 4),
        "scored_positions": "every token of the evaluation split",
        "next_token_positions": eval_tokens - eval_rows,
        "note": ("next-token prediction scores every position after the first of each "
                 "sequence; a context-free model scores the same either way"),
    }
    print(json.dumps(result, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
