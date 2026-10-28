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

Matching the corpus side as written undercounts badly -- only 9% of the corpus's own
SMILES are already in RDKit's canonical form -- so --canonical-corpus canonicalises every
distinct molecule-shaped string in the corpus across a process pool and answers the
molecule question on that set instead. The sampled canonical share is reported either way,
because it is what says how much the written-form count is worth.

--pair-index-out writes which pairs matched, by their index in the test-pair order,
which is the order scripts/molnl_retrieval_eval.py numbers its queries in. That is what
lets an already scored retrieval measurement be split into the pairs the corpus holds and
the pairs it does not, without scoring anything again.

    python scripts/molnl_downstream_pair_overlap.py --dataset <packed corpus> \
        --pairs-root <Mol-Instructions dir> --out <json> --pair-index-out <json>
"""

from __future__ import annotations

import argparse
import json
import random
import re
from multiprocessing import Pool

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


def _canonical_chunk(texts):
    """Canonicalise a chunk of written molecules. RDKit is imported per worker."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for text in texts:
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            out.append(Chem.MolToSmiles(mol))
    return out


def canonical_corpus(strings, processes, chunk=20_000):
    """The corpus's molecules as canonical SMILES. Names and formulas drop out here."""
    chunks = [strings[i:i + chunk] for i in range(0, len(strings), chunk)]
    seen = set()
    with Pool(processes) as pool:
        for i, got in enumerate(pool.imap_unordered(_canonical_chunk, chunks), 1):
            seen.update(got)
            if i % 20 == 0:
                print(f"    canonicalised {i * chunk:,} of {len(strings):,} "
                      f"-> {len(seen):,} molecules", flush=True)
    return seen


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
    ap.add_argument("--canonical-corpus", action="store_true",
                    help="canonicalise every corpus molecule instead of matching as written")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--pair-index-out",
                    help="JSON listing which test pairs matched, by their index in the "
                         "test-pair order, so an already scored measurement can be split "
                         "into the pairs the corpus contains and the pairs it does not")
    ap.add_argument("--show-records", type=int, default=0,
                    help="print this many corpus records as read, before splitting, so "
                         "the record form a measurement imitates is in the run's own log")
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
    if a.show_records:
        text = tokenizer.decode(data[a.split][0][a.column])
        for chunk in text.split("<|endoftext|>")[:a.show_records]:
            print(f"  record as read: {chunk + '<|endoftext|>'!r}", flush=True)

    seen_mol, seen_desc = set(), set()
    pair_hits = {}
    written = set() if a.canonical_corpus else None
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
            if written is not None and side and " " not in side:
                written.add(side)
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

    molecules_canonical = None
    if written is not None:
        print(f"  molecule-shaped strings: {len(written):,} distinct", flush=True)
        corpus_mols = canonical_corpus(sorted(written), a.processes)
        molecules_canonical = len(wanted_mol & corpus_mols)
        print(f"  分子（両側を正規化して照合）: {molecules_canonical:,} / {len(wanted_mol):,} "
              f"（コーパス側 {len(corpus_mols):,} 分子）")

    # A pair counts as trained on when the record carrying the description also carries
    # the molecule -- compared after canonicalising the record's own side.
    pairs_in_one_record = 0
    paired_at = []
    for i, (mol_text, desc) in enumerate(pairs):
        target = canon.get(mol_text)
        if not target or desc not in pair_hits:
            continue
        for other in pair_hits[desc]:
            mol = Chem.MolFromSmiles(other)
            if mol is not None and Chem.MolToSmiles(mol) == target:
                pairs_in_one_record += 1
                paired_at.append(i)
                break

    out = {
        "subset": a.subset, "split": a.split, "pairs": len(pairs),
        "selfies_undecodable": undecodable,
        "distinct_molecules": len(wanted_mol), "distinct_descriptions": len(wanted_desc),
        "molecules_in_corpus_as_written": len(seen_mol),
        "molecules_in_corpus_canonical": molecules_canonical,
        "corpus_distinct_molecules": len(corpus_mols) if molecules_canonical is not None else None,
        "descriptions_in_corpus": len(seen_desc),
        "pairs_in_one_record": pairs_in_one_record,
        "corpus_records": total, "corpus_records_response_with_space": nl_like,
        "corpus_smiles_sampled": ok, "corpus_smiles_already_canonical": same,
        "corpus_smiles_canonical_share": canonical_share,
    }
    print(f"  分子（コーパス側は書かれたまま）: {len(seen_mol):,} / {len(wanted_mol):,}")
    print(f"  説明文が完全一致で在る: {len(seen_desc):,} / {len(wanted_desc):,}")
    print(f"  同一レコードに対で在る: {pairs_in_one_record:,} / {len(pairs):,}")
    if canonical_share is not None:
        print(f"  コーパス SMILES のうち既に正規形: {same:,}/{ok:,} = {canonical_share:.3f}")
    if a.pair_index_out:
        index = {
            "subset": a.subset, "split": a.split, "pairs": len(pairs),
            "order": "the test-pair order scripts/molnl_retrieval_eval.py scores in",
            "pairs_in_one_record": paired_at,
            "description_in_corpus": [i for i, (_, d) in enumerate(pairs)
                                      if d in seen_desc],
            "molecule_in_corpus_as_written": [i for i, (m, _) in enumerate(pairs)
                                              if canon.get(m) in seen_mol],
        }
        json.dump(index, open(a.pair_index_out, "w"), indent=2)
        print(f"  wrote {a.pair_index_out} "
              f"({len(paired_at)} pair indices)")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
