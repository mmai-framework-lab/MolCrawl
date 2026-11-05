#!/usr/bin/env python3
"""Freeze the molecule/description pairs of the SMolInstruct test split.

The 2026-11-04d judgment moves the retrieval evaluation off zjunlp/Mol-Instructions and
onto the test split of the corpus's own distribution. Two things about that split decide
how the pairs have to be taken, and both are easy to get wrong:

  one pair, two tasks   molecule_captioning asks for the description of a molecule and
                        molecule_generation asks for the molecule of a description. They
                        are the same pairs in opposite directions -- in train and valid
                        the two counts are equal to the record (56,498 and 1,269) -- and
                        the retrieval evaluation already derives both directions from one
                        pair list. Taking pairs from both tasks would score each pair
                        twice and the two directions would stop being independent, so
                        only the captioning rows are read here.
  the molecule as written
                        the split holds the molecule in the spelling the corpus uses, of
                        which only 9.0% is RDKit-canonical. Canonicalising it would
                        rebuild the very notation mismatch this move is meant to remove,
                        so the molecule is carried through untouched.

The split has no column holding the response as text -- only output_ids -- so the
description is decoded with the corpus's own GPT-2 BPE.

The pairs are written once, to a file, and both the scoring and the overlap check read
that file. The earlier round measured 1,000 pairs in one direction and 250 in the other
and canonicalised one side only; a frozen pair list is what stops that recurring.

    python scripts/molnl_smolinstruct_test_pairs.py --arrow-root <arrow_splits> \
        --out <jsonl> --counts-out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import random

CAPTIONING = "molecule_captioning"
GENERATION = "molecule_generation"


def load_tokenizer():
    """The corpus's GPT-2 BPE, from GPT2_TOKENIZER_DIR -- no download on a compute node."""
    from transformers import AutoTokenizer

    where = os.environ.get("GPT2_TOKENIZER_DIR")
    if not where:
        raise SystemExit("set GPT2_TOKENIZER_DIR to a directory holding the GPT-2 tokenizer")
    return AutoTokenizer.from_pretrained(where)


def rows_of(dataset, task):
    """The rows of one task, with the two validity flags counted rather than assumed."""
    kept, too_long, invalid = [], 0, 0
    for row in dataset:
        if row["task_type"] != task:
            continue
        if row.get("input_too_long"):
            too_long += 1
            continue
        if row.get("valid_sample") is False:
            invalid += 1
            continue
        kept.append(row)
    return kept, too_long, invalid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arrow-root", required=True,
                    help="the directory holding test.arrow")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs", type=int, default=1000,
                    help="how many pairs to draw; fewer are written when fewer survive")
    ap.add_argument("--seed", type=int, default=42,
                    help="the seed the draw is made with, recorded in the counts file")
    ap.add_argument("--out", required=True, help="JSON Lines, one pair per line")
    ap.add_argument("--counts-out")
    a = ap.parse_args()

    from datasets import load_from_disk

    data = load_from_disk(os.path.join(a.arrow_root, f"{a.split}.arrow"))
    tokenizer = load_tokenizer()
    print(f"  {a.split}: {len(data):,} records", flush=True)

    kept, too_long, invalid = rows_of(data, CAPTIONING)
    print(f"  {CAPTIONING}: {len(kept):,} usable "
          f"(dropped input_too_long {too_long:,}, valid_sample False {invalid:,})",
          flush=True)

    pairs, seen = [], set()
    duplicates = 0
    for row in kept:
        molecule = row["input_text"].strip()
        description = tokenizer.decode(row["output_ids"]).split("<|endoftext|>")[0].strip()
        if not molecule or not description:
            continue
        key = (molecule, description)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        pairs.append({"molecule": molecule, "description": description})
    print(f"  distinct pairs: {len(pairs):,} (dropped {duplicates:,} repeats)", flush=True)

    # Recorded, not acted on: it is the evidence for reading one task rather than both.
    gen, gen_long, gen_invalid = rows_of(data, GENERATION)
    gen_pairs = {(r["output_text"].strip() if "output_text" in r else
                  tokenizer.decode(r["output_ids"]).split("<|endoftext|>")[0].strip(),
                  r["input_text"].strip()) for r in gen}
    shared = sum(1 for p in pairs if (p["molecule"], p["description"]) in gen_pairs)
    print(f"  {GENERATION}: {len(gen):,} usable; {shared:,} of the captioning pairs "
          f"appear there with the sides swapped", flush=True)

    rng = random.Random(a.seed)
    drawn = pairs if len(pairs) <= a.pairs else rng.sample(pairs, a.pairs)
    # Sorted after sampling so the file is stable to read; the draw itself is the seed's.
    drawn.sort(key=lambda p: (p["molecule"], p["description"]))
    with open(a.out, "w", encoding="utf-8") as fh:
        for p in drawn:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  wrote {a.out} ({len(drawn):,} pairs)")

    counts = {
        "split": a.split, "records_in_split": len(data),
        "task": CAPTIONING, "usable": len(kept),
        "dropped_input_too_long": too_long, "dropped_valid_sample_false": invalid,
        "distinct_pairs": len(pairs), "dropped_repeats": duplicates,
        "generation_rows_usable": len(gen),
        "generation_dropped_input_too_long": gen_long,
        "generation_dropped_valid_sample_false": gen_invalid,
        "captioning_pairs_also_in_generation": shared,
        "asked_for": a.pairs, "drawn": len(drawn), "seed": a.seed,
        "draw": ("every usable pair (fewer than asked for)" if len(pairs) <= a.pairs
                 else f"random.Random({a.seed}).sample over the distinct pairs"),
        "molecule": "as written in the split, not canonicalised",
        "description": "decoded from output_ids with the corpus's GPT-2 BPE",
    }
    if a.counts_out:
        json.dump(counts, open(a.counts_out, "w"), ensure_ascii=False, indent=1)
        print(f"  wrote {a.counts_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
