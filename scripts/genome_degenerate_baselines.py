"""Per-subset degenerate baselines for genome GPT-2 and BERT.

The degenerate baseline is the loss a model reaches by ignoring context and
predicting the marginal base distribution of the split it is scored on. It is
the line below which a number means "learned something" and above which it does
not. 1.3703 has been carried as a single figure for the whole campaign; it was
computed on one subset, so it says nothing about how predictable any other
subset's valid split is on its own.

Both models exclude the same ambiguous tokens from the loss -- genome's vocab is
A/T/G/C/N plus specials, so of the eleven IUPAC codes only N resolves, and the
rest fall to [UNK]. The baseline is therefore the entropy of the distribution
over the positions that actually carry loss, which is what this computes rather
than assuming a uniform ln(4).
"""
import argparse
import json
import math
import os
from collections import Counter

# genome single-nucleotide vocab
ID = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4,
      "[PAD]": 5, "[UNK]": 6, "[CLS]": 7, "[SEP]": 8, "[MASK]": 9}
NAME = {v: k for k, v in ID.items()}
STRUCTURAL = {ID["[PAD]"], ID["[CLS]"], ID["[SEP]"], ID["[MASK]"]}
AMBIGUOUS = {ID["N"]}          # the only IUPAC code this vocab resolves


def entropy(counts):
    """Natural-log entropy of a count distribution -- the cross-entropy a
    marginal predictor pays, in the same units the trainer reports loss."""
    total = sum(counts.values())
    if not total:
        return None, 0
    h = 0.0
    for c in counts.values():
        if c:
            p = c / total
            h -= p * math.log(p)
    return h, total


def tally(split, rows, chunk):
    """Token counts over a split, read in batches to bound memory."""
    counts = Counter()
    n = len(split)
    take = range(n) if rows is None or rows >= n else range(rows)
    idx = list(take)
    for lo in range(0, len(idx), chunk):
        for row in split[idx[lo:lo + chunk]]["input_ids"]:
            counts.update(row)
    return counts, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--bert-root", required=True,
                    help="root holding the derived 1,026-token BERT builds")
    ap.add_argument("--subsets", default="")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--rows", type=int, default=None,
                    help="rows per split; default every row")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from datasets import load_from_disk

    subsets = ([s for s in args.subsets.split(",") if s] or
               sorted(d for d in os.listdir(args.bert_root)
                      if os.path.isdir(os.path.join(args.bert_root, d,
                                                    "training_ready_hf_dataset_bert"))))
    results = []
    for s in subsets:
        print(f"\n########## {s} ##########", flush=True)
        row = {"subset": s, "split": args.split}
        for model, root, sub in (("gpt2", args.src_root, "training_ready_hf_dataset_gpt2"),
                                 ("bert", args.bert_root, "training_ready_hf_dataset_bert")):
            path = os.path.join(root, s, sub)
            ds = load_from_disk(path)[args.split]
            counts, used = tally(ds, args.rows, args.chunk)

            scored = Counter({k: v for k, v in counts.items()
                              if k not in STRUCTURAL and k not in AMBIGUOUS})
            h, total = entropy(scored)
            gc = sum(scored[ID[b]] for b in "GC") / total if total else float("nan")
            amb = sum(counts[a] for a in AMBIGUOUS)
            unk = counts.get(ID["[UNK]"], 0)

            row[model] = {
                "rows_used": used,
                "scored_tokens": total,
                "baseline": h,
                "gc_fraction": gc,
                "excluded_ambiguous": amb,
                "unk_tokens": unk,
                "composition": {NAME[k]: v / total for k, v in sorted(scored.items())},
            }
            print(f"  {model:5s} rows {used:>7,}  scored {total:>13,}"
                  f"  baseline {h:.4f}  GC {100*gc:.2f}%"
                  f"  N excluded {amb:,}  UNK {unk:,}", flush=True)

        d = row["bert"]["baseline"] - row["gpt2"]["baseline"]
        print(f"  bert - gpt2 = {d:+.4f}   (ln 4 = {math.log(4):.4f})", flush=True)
        results.append(row)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
