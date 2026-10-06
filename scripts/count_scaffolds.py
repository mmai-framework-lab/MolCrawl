"""Count distinct Bemis-Murcko scaffolds in a SMILES corpus.

Molecule count says how much data there is; scaffold count says how much of it is
different. ZINC-22 is 2,800 times organix13 by molecules, and the two numbers do
not scale together -- the order asks for both whenever a data-volume claim is
made, which is why this exists separately from the token count.

The scaffold is taken exactly as the downstream evaluation takes it
(tasks/evaluation/chembl_scaffold_heldout/splits.py): MurckoScaffoldSmiles with
includeChirality=False. A different definition here would make the overlap figure
incomparable with the split it is meant to be checked against.

Molecules whose scaffold is empty are counted separately rather than folded into
the distinct count: an acyclic molecule has no ring system, so the empty string is
the absence of a scaffold, not a scaffold that many molecules share.

Reads the parquet in batches and never holds the corpus in memory; the scaffold
set is what grows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool


def _scaffolds(smiles_batch):
    """(scaffolds, n_unparsed, n_acyclic, n_molecules) for one batch. Runs in a worker.

    The batch length comes back with the rest because imap_unordered does not keep
    the inputs: counting the rows in a second pass would read the whole parquet
    again for a number the worker already has.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    out, bad, acyclic = set(), 0, 0
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad += 1
            continue
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:                       # noqa: BLE001 - counted, not raised
            bad += 1
            continue
        if s:
            out.add(s)
        else:
            acyclic += 1
    return out, bad, acyclic, len(smiles_batch)


def batches(path, column, size, limit=None):
    import pyarrow.parquet as pq

    seen = 0
    for rb in pq.ParquetFile(path).iter_batches(batch_size=size, columns=[column]):
        vals = [v for v in rb.column(column).to_pylist() if v]
        if limit is not None and seen + len(vals) > limit:
            vals = vals[: limit - seen]
        seen += len(vals)
        if vals:
            yield vals
        if limit is not None and seen >= limit:
            return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--column", default="smiles")
    ap.add_argument("--batch-size", type=int, default=20000)
    ap.add_argument("--processes", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=None, help="stop after this many molecules")
    ap.add_argument("--out", help="write the counts here as JSON")
    args = ap.parse_args(argv)

    scaffolds, molecules, bad, acyclic = set(), 0, 0, 0
    with Pool(args.processes) as pool:
        for s, b, a, n in pool.imap_unordered(
            _scaffolds, batches(args.parquet, args.column, args.batch_size, args.limit)
        ):
            scaffolds |= s
            bad += b
            acyclic += a
            molecules += n

    result = {
        "parquet": args.parquet,
        "molecules": molecules,
        "scaffolds": len(scaffolds),
        "unparsed": bad,
        "acyclic": acyclic,
        "molecules_per_scaffold": round(molecules / len(scaffolds), 2) if scaffolds else None,
    }
    for k, v in result.items():
        print(f"  {k:<24}{v:,}" if isinstance(v, int) else f"  {k:<24}{v}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
