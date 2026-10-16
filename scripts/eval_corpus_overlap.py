"""How much of a downstream evaluation set is already in the training corpus.

An evaluation measures generalisation only to the extent its molecules were not
trained on, so the overlap has to be counted before the numbers mean anything. Two
overlaps are counted and they answer different questions: the same molecule appearing
in both is leakage, while the same Bemis-Murcko scaffold appearing in both is the
weaker statement that the shape was seen.

The evaluation sets are small enough to hold in memory, so the corpus is streamed past
them rather than the other way round. Both sides are canonicalised with RDKit first,
because two files naming one molecule rarely spell it the same way.

Scaffolds are computed for the evaluation sets and for a corpus only when asked:
Murcko decomposition costs far more than a hash lookup, which is affordable over
13 million molecules and not over 34 billion.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os


def canonical(smiles, scaffolds=False):
    """Canonical SMILES, and the Murcko scaffold when asked. None if unparsable."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol)
    if not scaffolds:
        return can, None
    try:
        # includeChirality=False matches tasks/evaluation/chembl_scaffold_heldout.
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:                               # noqa: BLE001 - counted as absent
        scaf = None
    return can, scaf


def load_eval(path):
    """Read the evaluation sets written by fetch_tdc_eval_sets.py's cache."""
    import pandas as pd

    df = pd.read_csv(path, sep="\t" if path.endswith((".tsv", ".tab")) else ",")
    col = next((c for c in ("Drug", "smiles", "SMILES", "Smiles") if c in df.columns), None)
    if col is None:
        raise SystemExit(f"{path} に SMILES 列が無い: {list(df.columns)}")
    return [str(x) for x in df[col].tolist()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-file", required=True, action="append",
                    help="an evaluation set; repeat for several")
    ap.add_argument("--corpus", required=True,
                    help="a .smi.gz, or a parquet with a SMILES column")
    ap.add_argument("--corpus-smiles-column", default="smiles")
    ap.add_argument("--smi-column", type=int, default=0)
    ap.add_argument("--scaffolds", action="store_true",
                    help="also count scaffold overlap; affordable only on a small corpus")
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)

    eval_mols, eval_scafs, per_set = {}, {}, {}
    for path in args.eval_file:
        name = os.path.basename(path)
        smis = load_eval(path)
        mols, scafs, bad = set(), set(), 0
        for smi in smis:
            can, scaf = canonical(smi, scaffolds=True)
            if can is None:
                bad += 1
                continue
            mols.add(can)
            if scaf:
                scafs.add(scaf)
        per_set[name] = {"rows": len(smis), "unique_molecules": len(mols),
                         "unique_scaffolds": len(scafs), "unparsable": bad,
                         "molecules_found": 0, "scaffolds_found": 0}
        for m in mols:
            eval_mols.setdefault(m, []).append(name)
        for s in scafs:
            eval_scafs.setdefault(s, []).append(name)
        print(f"  {name}: {len(smis):,} 行 → 分子 {len(mols):,} / 骨格 {len(scafs):,} "
              f"/ 解析不能 {bad}", flush=True)

    print(f"\n照合対象 分子 {len(eval_mols):,} / 骨格 {len(eval_scafs):,}", flush=True)
    print(f"corpus を流し読み: {args.corpus}", flush=True)

    hit_m, hit_s = set(), set()
    n = 0
    if args.corpus.endswith(".parquet"):
        import pyarrow.parquet as pq
        table = pq.read_table(args.corpus, columns=[args.corpus_smiles_column])
        source = (str(x) for x in table.column(args.corpus_smiles_column).to_pylist())
    else:
        def gen():
            with gzip.open(args.corpus, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) > args.smi_column:
                        yield parts[args.smi_column]
        source = gen()

    for smi in source:
        can, scaf = canonical(smi, scaffolds=args.scaffolds)
        if can is None:
            continue
        n += 1
        if can in eval_mols:
            hit_m.add(can)
        if args.scaffolds and scaf and scaf in eval_scafs:
            hit_s.add(scaf)
        if n % 1_000_000 == 0:
            print(f"    {n:,} 分子  分子一致 {len(hit_m):,}  骨格一致 {len(hit_s):,}", flush=True)

    for m in hit_m:
        for name in eval_mols[m]:
            per_set[name]["molecules_found"] += 1
    for s in hit_s:
        for name in eval_scafs[s]:
            per_set[name]["scaffolds_found"] += 1

    print(f"\ncorpus {n:,} 分子を読んだ")
    for name, v in per_set.items():
        mp = v["molecules_found"] / v["unique_molecules"] * 100 if v["unique_molecules"] else 0
        line = (f"  {name}: 分子 {v['molecules_found']:,}/{v['unique_molecules']:,} "
                f"({mp:.2f}%)")
        if args.scaffolds:
            sp = v["scaffolds_found"] / v["unique_scaffolds"] * 100 if v["unique_scaffolds"] else 0
            line += f"  骨格 {v['scaffolds_found']:,}/{v['unique_scaffolds']:,} ({sp:.2f}%)"
        print(line)

    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump({"corpus": os.path.abspath(args.corpus), "corpus_molecules_read": n,
                   "scaffolds_counted": bool(args.scaffolds), "sets": per_set}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\n報告 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
