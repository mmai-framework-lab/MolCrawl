"""Measure how much of a downstream evaluation set is already in the pretraining corpus.

The compounds corpus is OrganiX13.parquet, which combine_all.py builds from ZINC20
(5M sample), ZINC QM9, OPV and ChEMBL. Several MoleculeNet tasks draw from the same
public sources, so a molecule can appear in both. Where it does, the downstream score
is not measuring generalisation to unseen structures.

Matching is on RDKit canonical SMILES, so formatting differences between the two
sources do not hide a match. Two keys are reported: the isomeric canonical SMILES
(same molecule, stereochemistry included) and the flat canonical SMILES with
stereochemistry stripped (same skeleton, different stereoisomer). InChIKey would also
fold tautomers together but costs several milliseconds per molecule, which over a
13M-row corpus is hours rather than minutes -- so exact and flat SMILES it is, and the
numbers are a lower bound on overlap.

Read-only. Builds the corpus key set in memory, so it streams the parquet rather than
loading it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10 ** 7)


def keys_for(smiles: str) -> tuple[str, str] | None:
    """Return (isomeric canonical SMILES, flat canonical SMILES), or None if unparsable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return (Chem.MolToSmiles(mol, isomericSmiles=True),
            Chem.MolToSmiles(mol, isomericSmiles=False))


def corpus_keys(parquet: Path, column: str, limit: int | None):
    """Stream the corpus and return (isomeric keys, flat keys, n_read, n_parsed)."""
    full, block = set(), set()
    n_read = n_parsed = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=16384, columns=[column]):
        for smi in batch.column(0).to_pylist():
            if not smi:
                continue
            n_read += 1
            pair = keys_for(smi)
            if pair:
                n_parsed += 1
                full.add(pair[0])
                block.add(pair[1])
            if limit and n_read >= limit:
                return full, block, n_read, n_parsed
        if n_read % 500000 < 16384:
            print(f"  ... {n_read:,} rows read", flush=True)
    return full, block, n_read, n_parsed


def eval_smiles(path: Path, column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if column not in (rows[0] if rows else {}):
        raise KeyError(f"{path} has no column {column!r}; columns are {list(rows[0])[:8]}")
    return [r[column] for r in rows if r.get(column)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True, help="pretraining corpus parquet")
    ap.add_argument("--parquet-column", default="smiles")
    ap.add_argument("--corpus-limit", type=int, default=None,
                    help="stop after this many corpus rows (omit to use all of them)")
    ap.add_argument("--eval-root", required=True, help="directory holding <task>/raw.csv")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="task:smiles_column pairs, e.g. bace:mol bbbp:smiles")
    ap.add_argument("--output", help="write the results here as JSON")
    args = ap.parse_args()

    print(f"building corpus keys from {args.parquet}", flush=True)
    full, block, n_read, n_parsed = corpus_keys(
        Path(args.parquet), args.parquet_column, args.corpus_limit)
    print(f"corpus: {n_read:,} rows read, {n_parsed:,} parsed, "
          f"{len(full):,} distinct molecules, {len(block):,} distinct flat structures")
    print()

    results = []
    for spec in args.tasks:
        task, _, column = spec.partition(":")
        path = Path(args.eval_root) / task / "raw.csv"
        if not path.exists():
            print(f"{task:10s} SKIP  {path} does not exist")
            continue
        try:
            smis = eval_smiles(path, column or "smiles")
        except KeyError as exc:
            print(f"{task:10s} ERROR {exc}")
            continue
        parsed = [p for p in (keys_for(s) for s in smis) if p]
        hit_full = sum(1 for iso, _ in parsed if iso in full)
        hit_block = sum(1 for _, flat in parsed if flat in block)
        row = {
            "task": task, "rows": len(smis), "parsed": len(parsed),
            "exact_matches": hit_full,
            "exact_fraction": hit_full / len(parsed) if parsed else None,
            "skeleton_matches": hit_block,
            "skeleton_fraction": hit_block / len(parsed) if parsed else None,
        }
        results.append(row)
        print(f"{task:10s} {len(parsed):6d} molecules  "
              f"同一分子 {hit_full:6d} ({row['exact_fraction']:6.2%})  "
              f"同一骨格 {hit_block:6d} ({row['skeleton_fraction']:6.2%})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "parquet": str(args.parquet),
            "corpus_rows_read": n_read, "corpus_keys": len(full),
            "corpus_limit": args.corpus_limit, "results": results,
        }, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
