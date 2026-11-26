#!/usr/bin/env python3
"""How much of ChEBI-20's test split the molnl pretraining corpus already holds.

The 2026-11-26 order, item 6: for each test pair, is it in the corpus as a pair, as the
description alone, or as the molecule alone. The corpus is the train split of the packed
build the pretraining runs read; its records are

    <s>[INST] <prompt> [/INST]<response><|endoftext|>

and are split back apart with the same reader scripts/molnl_downstream_pair_overlap.py
uses. What counts as a match:

  molecule      RDKit's canonical isomeric SMILES of the test molecule equals RDKit's
                canonical isomeric SMILES of one whole side of a record. Every distinct
                space-free side of the corpus is canonicalised; a side RDKit will not
                read is a name or a formula and cannot match. A test molecule RDKit will
                not read is counted and left unmatched.
  description   one whole side of a record, whitespace collapsed, equals the test
                description, whitespace collapsed. Two spellings of the description are
                tried, because ChEBI-20 is published twice: Text2Mol's, which opens with
                the molecule's name and is what the retrieval task reads, and MolT5's,
                which replaces the name with "The molecule". Which one matched is kept.
  pair          a single record has the description on one side and the molecule on the
                other.

The four classes are exclusive: pair; description and molecule both present but never in
one record; description only; molecule only. Anything else is "neither".

    python scripts/molnl_chebi20_overlap.py --dataset <packed corpus> \
        --chebi20 <table from molnl_chebi20_build.py> --out-tsv <per-item tsv> \
        --out-json <counts>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from molnl_downstream_pair_overlap import canonical_corpus, records  # noqa: E402

WS = re.compile(r"\s+")


def norm(text):
    return WS.sub(" ", text).strip()


def canon(smiles, chem):
    mol = chem.MolFromSmiles(smiles) if smiles else None
    return chem.MolToSmiles(mol) if mol is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="the packed molnl corpus (load_from_disk)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--column", default="input_ids")
    ap.add_argument("--chebi20", required=True)
    ap.add_argument("--which", default="test", help="ChEBI-20 split to check")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--show-records", type=int, default=5)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    from datasets import load_from_disk
    from rdkit import Chem, RDLogger

    from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

    RDLogger.DisableLog("rdApp.*")
    tokenizer = MoleculeNatLangTokenizer().tokenizer

    items = [json.loads(line) for line in open(a.chebi20, encoding="utf-8") if line.strip()]
    items = [r for r in items if r["split"] == a.which]
    print(f"  ChEBI-20 {a.which}: {len(items):,} pairs", flush=True)
    for r in items:
        r["canon"] = canon(r["smiles"], Chem)
    unread = sum(1 for r in items if r["canon"] is None)
    print(f"  test molecules RDKit would not canonicalise: {unread}", flush=True)

    want_desc = {}
    for r in items:
        want_desc.setdefault(norm(r["description"]), []).append((r["cid"], "text2mol"))
        if r["molt5_description"]:
            want_desc.setdefault(norm(r["molt5_description"]), []).append((r["cid"], "molt5"))

    data = load_from_disk(a.dataset)
    if a.show_records:
        text = tokenizer.decode(data[a.split][0][a.column])
        for chunk in text.split("<|endoftext|>")[:a.show_records]:
            print(f"  record as read: {chunk + '<|endoftext|>'!r}", flush=True)

    written = set()
    desc_hits = {}      # cid -> {variant: [other side, ...]}
    total = 0
    for prompt, response in records(data, a.split, a.column, tokenizer):
        total += 1
        for side, other in ((prompt, response), (response, prompt)):
            if side and " " not in side:
                written.add(side)
            key = norm(side)
            if key in want_desc:
                for cid, variant in want_desc[key]:
                    desc_hits.setdefault(cid, {}).setdefault(variant, []).append(other)
        if total % 500_000 == 0:
            print(f"    records read: {total:,}", flush=True)
    print(f"  corpus records: {total:,}; distinct space-free sides: {len(written):,}",
          flush=True)

    corpus_mols = canonical_corpus(sorted(written), a.processes)
    print(f"  corpus molecules after canonicalisation: {len(corpus_mols):,}", flush=True)

    rows, classes = [], Counter()
    variant_counts = Counter()
    for r in items:
        hits = desc_hits.get(r["cid"], {})
        desc_in = bool(hits)
        mol_in = r["canon"] is not None and r["canon"] in corpus_mols
        paired_variants = []
        for variant, others in hits.items():
            if r["canon"] and any(canon(o, Chem) == r["canon"] for o in others
                                  if o and " " not in o):
                paired_variants.append(variant)
        if paired_variants:
            cls = "pair"
        elif desc_in and mol_in:
            cls = "description_and_molecule_apart"
        elif desc_in:
            cls = "description_only"
        elif mol_in:
            cls = "molecule_only"
        else:
            cls = "neither"
        classes[cls] += 1
        for v in hits:
            variant_counts[v] += 1
        rows.append({
            "cid": r["cid"], "class": cls,
            "description_matched_as": ",".join(sorted(hits)) or "-",
            "description_records": sum(len(v) for v in hits.values()),
            "molecule_in_corpus": "yes" if mol_in else ("unreadable" if r["canon"] is None else "no"),
            "pair_matched_as": ",".join(sorted(paired_variants)) or "-",
            "smiles_canonical": r["canon"] or "",
        })

    with open(a.out_tsv, "w", encoding="utf-8") as fh:
        cols = list(rows[0])
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")

    out = {
        "chebi20_split": a.which, "pairs": len(items),
        "test_molecules_not_canonicalised": unread,
        "corpus_split": a.split, "corpus_records": total,
        "corpus_space_free_sides": len(written),
        "corpus_molecules_canonical": len(corpus_mols),
        "classes": dict(classes),
        "description_found_by_variant": dict(variant_counts),
        "molecule_in_corpus": sum(1 for row in rows if row["molecule_in_corpus"] == "yes"),
        "description_in_corpus": sum(1 for row in rows if row["description_matched_as"] != "-"),
    }
    print(json.dumps(out, indent=1))
    with open(a.out_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
