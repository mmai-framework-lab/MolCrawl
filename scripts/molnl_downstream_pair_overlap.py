#!/usr/bin/env python3
"""Does the pretraining corpus contain the downstream test pairs themselves?

The window-hash check (scripts/molnl_downstream_contamination.py) cannot answer this
for the molecule side: the corpus writes molecules as SMILES while Mol-Instructions
writes them as SELFIES, so a molecule that was pretrained on scores zero overlap. It
is a difference of notation, not of content, and reading it as "clean" would be wrong.

So compare the records instead. The corpus is packed as

    <s>[INST] <molecule or name> [/INST]<molecule, formula, name or description><eot>

which is split back apart here. For each of the test pairs:

  description   does the exact string occur as a record's response?
  molecule      does its canonical SMILES (SELFIES -> SMILES -> RDKit canonical) occur
                anywhere in the corpus, on either side of a record?
  the pair      do both occur in the *same* record -- the only reading under which the
                downstream task itself was trained on.

Because the corpus side is matched as written, the fraction of corpus molecules that
are already RDKit-canonical is measured on a sample and reported with the result: it
bounds how much the molecule count can be an undercount.

    python scripts/molnl_downstream_pair_overlap.py --dataset <packed corpus> \
        --pairs-root <Mol-Instructions dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
import random
import re

RECORD = re.compile(r"\[INST\](.*?)\[/INST\](.*)", re.S)


def test_pairs(root, subset):
    import os

    from datasets import load_from_disk

    data = load_from_disk(os.path.join(root, subset))
    rows = []
    for item in data:
        meta = item["metadata"]
        if not isinstance(meta, dict):
            meta = json.loads(str(meta).replace("'", '"'))
        if str(meta.get("split", "")).lower() == "test":
            rows.append((item["input"].strip(), item["output"].strip()))
    return rows


def canonical(selfies_text, selfies_mod, chem):
    """SELFIES -> canonical SMILES, or None when either step refuses the string."""
    try:
        smiles = selfies_mod.decoder(selfies_text)
    except Exception:
        return None
    if not smiles:
        return None
    mol = chem.MolFromSmiles(smiles)
    return chem.MolToSmiles(mol) if mol is not None else None


def records(dataset, split, column, tokenizer):
    """Yield (prompt, response) over the packed corpus, across row boundaries."""
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
    ap.add_argument("--pairs-root", required=True)
    ap.add_argument("--subset", default="molecular_description_generation")
    ap.add_argument("--split", default="train")
    ap.add_argument("--column", default="input_ids")
    ap.add_argument("--sample", type=int, default=100_000,
                    help="corpus molecules canonicality is measured on this many")
    ap.add_argument("--out")
    a = ap.parse_args()

    import selfies as selfies_mod
    from datasets import load_from_disk
    from rdkit import Chem, RDLogger

    from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

    RDLogger.DisableLog("rdApp.*")
    tokenizer = MoleculeNatLangTokenizer().tokenizer
    pairs = test_pairs(a.pairs_root, a.subset)
    print(f"  test pairs: {len(pairs):,}", flush=True)

    canon = {}
    for mol_text, _ in pairs:
        canon[mol_text] = canonical(mol_text, selfies_mod, Chem)
    undecodable = sum(1 for v in canon.values() if v is None)
    wanted_mol = {v for v in canon.values() if v}
    wanted_desc = {d for _, d in pairs}
    print(f"  SELFIES -> canonical SMILES: {len(wanted_mol):,} distinct, "
          f"{undecodable} could not be decoded", flush=True)

    data = load_from_disk(a.dataset)
    seen_mol, seen_desc = set(), set()
    pair_hits = {}
    total, nl_like, sampled = 0, 0, []
    rng = random.Random(42)
    for prompt, response in records(data, a.split, a.column, tokenizer):
        total += 1
        if " " in response:
            nl_like += 1
        for side in (prompt, response):
            if side in wanted_mol:
                seen_mol.add(side)
            if side in wanted_desc:
                seen_desc.add(side)
        if response in wanted_desc or prompt in wanted_desc:
            desc = response if response in wanted_desc else prompt
            other = prompt if response in wanted_desc else response
            pair_hits.setdefault(desc, set()).add(other)
        if " " not in prompt and len(sampled) < a.sample:
            sampled.append(prompt)
        elif " " not in prompt and rng.random() < a.sample / max(total, 1):
            sampled[rng.randrange(a.sample)] = prompt
    print(f"  corpus records: {total:,} ({nl_like:,} whose response contains a space)",
          flush=True)

    # How much can matching the corpus side as written undercount? Measure it.
    ok, same = 0, 0
    for text in sampled:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        ok += 1
        if Chem.MolToSmiles(mol) == text:
            same += 1
    canonical_share = (same / ok) if ok else None

    # A pair counts as trained on when the record carrying the description also carries
    # the molecule -- compared after canonicalising the record's own side.
    pairs_in_one_record = 0
    for mol_text, desc in pairs:
        target = canon.get(mol_text)
        if not target or desc not in pair_hits:
            continue
        for other in pair_hits[desc]:
            mol = Chem.MolFromSmiles(other)
            if mol is not None and Chem.MolToSmiles(mol) == target:
                pairs_in_one_record += 1
                break

    out = {
        "subset": a.subset, "split": a.split, "pairs": len(pairs),
        "selfies_undecodable": undecodable,
        "distinct_molecules": len(wanted_mol), "distinct_descriptions": len(wanted_desc),
        "molecules_in_corpus": len(seen_mol),
        "descriptions_in_corpus": len(seen_desc),
        "pairs_in_one_record": pairs_in_one_record,
        "corpus_records": total, "corpus_records_response_with_space": nl_like,
        "corpus_smiles_sampled": ok, "corpus_smiles_already_canonical": same,
        "corpus_smiles_canonical_share": canonical_share,
    }
    print(f"  分子（正規化 SMILES）がコーパスに在る: {len(seen_mol):,} / {len(wanted_mol):,}")
    print(f"  説明文が完全一致で在る: {len(seen_desc):,} / {len(wanted_desc):,}")
    print(f"  同一レコードに対で在る: {pairs_in_one_record:,} / {len(pairs):,}")
    if canonical_share is not None:
        print(f"  コーパス SMILES のうち既に正規形: {same:,}/{ok:,} = {canonical_share:.3f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
