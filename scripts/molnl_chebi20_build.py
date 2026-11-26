#!/usr/bin/env python3
"""One table of ChEBI-20 as Text2Mol distributes it: 33,010 molecule-description pairs.

The 2026-11-26 order asks for ChEBI-20 as distributed, its own splits, and all 33,010
molecules as the candidate set. Two releases go by that name and they differ:

  Text2Mol (cnedwards/text2mol, data/)   training.txt / val.txt / test.txt carry
      CID, a mol2vec vector and the description; training/val/test.sdf.zip carry the
      structures. 26,408 / 3,301 / 3,301 = 33,010. The descriptions open with the
      molecule's own name ("Glyceollin IV is a member of ...").
  MolT5 (blender-nlp/MolT5, ChEBI-20_data/)   CID, SMILES, description.
      26,407 / 3,301 / 3,300 = 33,008: CIDs 135476785 (copernicium atom, train) and
      135476786 (roentgenium atom, test) are absent. The descriptions have the name
      replaced by "The molecule" ("The molecule is a member of ...").

The order's 33,010 and the scores it compares against (Text2Mol's MRR 0.372 / 0.499)
are Text2Mol's, so this table is Text2Mol's release: its splits, its descriptions, and a
SMILES written by RDKit from its own SDF. MolT5's SMILES and description for the same
CID are carried alongside, for the overlap check and for anyone who wants the masked
text, but nothing downstream reads them as the pair.

    python scripts/molnl_chebi20_build.py --text2mol <dir with *.txt and sdf/> \
        --molt5 <dir with train/validation/test.txt> --out <jsonl>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

SPLITS = (("train", "training"), ("valid", "val"), ("test", "test"))
MOLT5_NAMES = {"train": "train", "valid": "validation", "test": "test"}


def read_text2mol_txt(path):
    """[(cid, description)] in file order. Columns: CID, mol2vec, description."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise SystemExit(f"{path}: expected 3 tab-separated fields, got {len(fields)}")
            out.append((fields[0].strip(), fields[2].strip()))
    return out


def read_sdf(path):
    """{cid: canonical isomeric SMILES or None}. The record's title line is the CID."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = {}
    supplier = Chem.SDMolSupplier(path, sanitize=True, removeHs=True)
    for i, mol in enumerate(supplier):
        if mol is None:
            # The title is lost with the molecule; recover it from the raw block.
            title = supplier.GetItemText(i).split("\n", 1)[0].strip()
            out[title] = None
            continue
        out[mol.GetProp("_Name").strip()] = Chem.MolToSmiles(mol)
    return out


def read_molt5(path):
    """{cid: (smiles, description)} from MolT5's tab-separated release (header row)."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        if header != ["CID", "SMILES", "description"]:
            raise SystemExit(f"{path}: unexpected header {header}")
        for row in reader:
            out[row[0].strip()] = (row[1].strip(), row[2].strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text2mol", required=True,
                    help="directory with training/val/test.txt and sdf/{training,val,test}.sdf")
    ap.add_argument("--molt5", required=True,
                    help="directory with MolT5's train/validation/test.txt")
    ap.add_argument("--out", required=True, help="output JSON lines, one pair per line")
    ap.add_argument("--summary", help="JSON with the counts this run measured")
    args = ap.parse_args()

    rows, counts = [], {}
    seen = set()
    for split, stem in SPLITS:
        pairs = read_text2mol_txt(os.path.join(args.text2mol, f"{stem}.txt"))
        structures = read_sdf(os.path.join(args.text2mol, "sdf", f"{stem}.sdf"))
        molt5 = read_molt5(os.path.join(args.molt5, f"{MOLT5_NAMES[split]}.txt"))
        c = Counter()
        c["rows"] = len(pairs)
        c["sdf_records"] = len(structures)
        for cid, desc in pairs:
            if cid in seen:
                raise SystemExit(f"CID {cid} appears in more than one split")
            seen.add(cid)
            if cid not in structures:
                c["no_sdf_record"] += 1
                smiles = ""
            elif structures[cid] is None:
                c["sdf_unparsed"] += 1
                smiles = ""
            else:
                smiles = structures[cid]
            m_smiles, m_desc = molt5.get(cid, ("", ""))
            c["in_molt5"] += cid in molt5
            rows.append({
                "cid": cid, "split": split, "smiles": smiles, "description": desc,
                "molt5_smiles": m_smiles, "molt5_description": m_desc,
            })
        c["molt5_rows"] = len(molt5)
        c["molt5_not_in_text2mol"] = len(set(molt5) - {cid for cid, _ in pairs})
        counts[split] = dict(c)

    dup = Counter(r["smiles"] for r in rows if r["smiles"])
    counts["all"] = {
        "rows": len(rows),
        "with_smiles": sum(1 for r in rows if r["smiles"]),
        "distinct_smiles": len(dup),
        "smiles_shared_by_more_than_one_cid": sum(1 for v in dup.values() if v > 1),
        "rows_whose_smiles_is_shared": sum(v for v in dup.values() if v > 1),
    }

    # JSON lines rather than TSV: SMILES carry backslashes (stereo bonds) and a TSV
    # writer would have to escape them, which is how MolT5's own file came to double them.
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(counts, indent=1))
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as fh:
            json.dump(counts, fh, indent=1)
    missing = sum(counts[s].get("no_sdf_record", 0) + counts[s].get("sdf_unparsed", 0)
                  for s, _ in SPLITS)
    if missing:
        print(f"{missing} rows have no SMILES", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
