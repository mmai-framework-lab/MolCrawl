#!/usr/bin/env python3
"""What the pretraining corpus is made of, counted by the shape of each record.

4.0% of the corpus is the description-generating form. The 2026-10-23 order asks what the
other 96.0% is. Each record is

    <s>[INST] <prompt> [/INST]<response><eot>

so both sides are classified and the pairs counted. The classes are decided in this
order, which is what makes them reproducible rather than a matter of taste:

  sentence     a space, at least three words, and an English function word among them.
               "contains a space" alone does not separate prose from a chemical name:
               "4-(acetamidomethyl)-1H-pyrazole-5-carboxylic acid" has a space and is a
               name, and counting those as prose put the corpus's share of description
               records at 4.0% when it is far smaller.
  text         a space but no sentence shape -- the chemical names that carry one.
  smiles       RDKit parses it. "CCO" is a molecule; "C18H20N6O2" is not one to RDKit,
               which is what separates a formula from a short SMILES.
  formula      element symbols with counts and nothing else.
  name         anything left that has a lowercase letter -- the IUPAC-style names.
  other        the remainder, reported rather than folded into a neighbour.

Distinct strings are classified once and the result reused, because the sides repeat.

    python scripts/molnl_corpus_record_types.py --dataset <packed corpus> --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from multiprocessing import Pool

RECORD = re.compile(r"\[INST\](.*?)\[/INST\](.*)", re.S)
FORMULA = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
TEXT, SMILES, FORMULA_T, NAME, OTHER = "text", "smiles", "formula", "name", "other"
SENTENCE = "sentence"
# Words that appear in prose and never inside a chemical identifier.
FUNCTION_WORDS = {"is", "are", "was", "were", "the", "a", "an", "of", "with", "and",
                  "in", "to", "has", "have", "it", "this", "that", "by", "from", "for"}


def _is_sentence(s):
    words = s.split()
    if len(words) < 3:
        return False
    return any(w.strip(".,;:()").lower() in FUNCTION_WORDS for w in words)


def _classify_chunk(strings):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in strings:
        if not s:
            out.append(OTHER)
        elif " " in s:
            out.append(SENTENCE if _is_sentence(s) else TEXT)
        elif Chem.MolFromSmiles(s) is not None:
            out.append(SMILES)
        elif FORMULA.match(s):
            out.append(FORMULA_T)
        elif any(c.islower() for c in s):
            out.append(NAME)
        else:
            out.append(OTHER)
    return out


def records(dataset, split, column, tokenizer):
    tail = ""
    for row in dataset[split]:
        tail += tokenizer.decode(row[column])
        parts = tail.split("<|endoftext|>")
        tail = parts.pop()
        for chunk in parts:
            m = RECORD.search(chunk)
            if m:
                yield m.group(1).strip(), m.group(2).strip()
    m = RECORD.search(tail)
    if m:
        yield m.group(1).strip(), m.group(2).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--column", default="input_ids")
    ap.add_argument("--processes", type=int, default=16)
    ap.add_argument("--examples", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()

    from datasets import load_from_disk

    from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

    tokenizer = MoleculeNatLangTokenizer().tokenizer
    data = load_from_disk(a.dataset)

    pairs = []
    distinct = set()
    for prompt, response in records(data, a.split, a.column, tokenizer):
        pairs.append((prompt, response))
        distinct.add(prompt)
        distinct.add(response)
    print(f"  records {len(pairs):,}, distinct sides {len(distinct):,}", flush=True)

    strings = sorted(distinct)
    chunk = 20_000
    chunks = [strings[i:i + chunk] for i in range(0, len(strings), chunk)]
    kinds = {}
    with Pool(a.processes) as pool:
        for i, got in enumerate(pool.imap(_classify_chunk, chunks)):
            for s, k in zip(chunks[i], got):
                kinds[s] = k
            if (i + 1) % 25 == 0:
                print(f"    classified {min((i + 1) * chunk, len(strings)):,}"
                      f" of {len(strings):,}", flush=True)

    counts = Counter((kinds[p], kinds[r]) for p, r in pairs)
    examples = {}
    for prompt, response in pairs:
        key = (kinds[prompt], kinds[response])
        got = examples.setdefault(key, [])
        if len(got) < a.examples:
            got.append({"prompt": prompt[:160], "response": response[:160]})

    total = len(pairs)
    out = {"records": total, "distinct_sides": len(distinct), "forms": []}
    print(f"\n  {'prompt -> response':<26}{'records':>12}{'share':>9}")
    for (pk, rk), n in counts.most_common():
        print(f"  {pk + ' -> ' + rk:<26}{n:>12,}{n / total:>8.2%}")
        out["forms"].append({"prompt": pk, "response": rk, "records": n,
                             "share": round(n / total, 6),
                             "examples": examples[(pk, rk)]})
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
