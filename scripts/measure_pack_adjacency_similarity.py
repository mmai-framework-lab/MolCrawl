"""Measure how similar the molecules that share a packed block are.

Packing concatenates the corpus and cuts every 1,024 tokens, so the order the
molecules arrive in decides who shares an attention window. GPT-2 attends causally
across the whole block with no document mask, so a molecule can look back at its
neighbours. If neighbours in source order are more alike than random pairs, that is
the shortcut the 2026-08-21 reshuffle removed -- and the size of the gap says how
much of the +0.05 loss change it explains.

Compares three pairings over the same molecules:

  source     consecutive rows in the input parquet   (what v3/v4-source packed)
  shuffled   consecutive rows after the seeded permutation (what ships now)
  random     independently drawn pairs               (the floor)

Tanimoto over Morgan fingerprints. Read-only; nothing is written back to the corpus.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")


def load_smiles(parquet: Path, column: str, limit: int) -> list[str]:
    """Read the first `limit` SMILES in parquet row order.

    Streamed rather than read whole: the corpus is ~13M rows, and pulling the entire
    column in to take the first 20,000 costs a gigabyte for nothing.
    """
    out: list[str] = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8192, columns=[column]):
        for smi in batch.column(0).to_pylist():
            if smi:
                out.append(smi)
                if len(out) >= limit:
                    return out
    return out


def fingerprints(smiles: list[str], radius: int, bits: int):
    """Return (fingerprints, kept_indices); molecules RDKit rejects are dropped."""
    fps, kept = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits))
        kept.append(i)
    return fps, kept


def pair_similarities(fps, pairs) -> np.ndarray:
    return np.array([DataStructs.TanimotoSimilarity(fps[a], fps[b]) for a, b in pairs])


def summarise(name: str, sims: np.ndarray) -> dict:
    return {
        "pairing": name,
        "pairs": int(sims.size),
        "mean": float(sims.mean()),
        "median": float(np.median(sims)),
        "p90": float(np.percentile(sims, 90)),
        "frac_above_0.4": float((sims > 0.4).mean()),
        "frac_above_0.7": float((sims > 0.7).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True, help="corpus parquet (OrganiX13.parquet)")
    ap.add_argument("--column", default="smiles", help="SMILES column name")
    ap.add_argument("--limit", type=int, default=20000, help="molecules to read, in row order")
    ap.add_argument("--pairs", type=int, default=5000, help="pairs to score per pairing")
    ap.add_argument("--pack-order-seed", type=int, default=43,
                    help="must match PACK_ORDER_SEED in the packing script")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--output", help="write the results here as JSON")
    args = ap.parse_args()

    smiles = load_smiles(Path(args.parquet), args.column, args.limit)
    print(f"read {len(smiles):,} SMILES from {args.parquet} (column {args.column!r})")

    fps, kept = fingerprints(smiles, args.radius, args.bits)
    n = len(fps)
    print(f"fingerprinted {n:,} of {len(smiles):,} ({len(smiles) - n} rejected by RDKit)")
    if n < 100:
        print("too few molecules to compare")
        return 1

    # `source` walks the fingerprints in the order they were read, which is parquet
    # order -- the order v3 and PACK_ORDER=source pack in.
    step = max(1, (n - 1) // args.pairs)
    src_pairs = [(i, i + 1) for i in range(0, n - 1, step)][: args.pairs]

    # `shuffled` applies the same permutation the packing script does, then walks the
    # result, so consecutive entries are the molecules that now share a block.
    perm = np.random.default_rng(args.pack_order_seed).permutation(n)
    shuf_pairs = [(int(perm[i]), int(perm[i + 1])) for i in range(0, n - 1, step)][: args.pairs]

    # `random` is the floor: unrelated molecules from the same corpus.
    rng = random.Random(args.pack_order_seed)
    rnd_pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(args.pairs)]
    rnd_pairs = [(a, b) for a, b in rnd_pairs if a != b]

    rows = [
        summarise("source", pair_similarities(fps, src_pairs)),
        summarise("shuffled", pair_similarities(fps, shuf_pairs)),
        summarise("random", pair_similarities(fps, rnd_pairs)),
    ]

    print()
    hdr = f"{'pairing':10s} {'pairs':>7s} {'mean':>8s} {'median':>8s} {'p90':>8s} {'>0.4':>8s} {'>0.7':>8s}"
    print(hdr)
    for r in rows:
        print(f"{r['pairing']:10s} {r['pairs']:7d} {r['mean']:8.4f} {r['median']:8.4f} "
              f"{r['p90']:8.4f} {r['frac_above_0.4']:8.4f} {r['frac_above_0.7']:8.4f}")

    src, shuf, rnd = rows[0]["mean"], rows[1]["mean"], rows[2]["mean"]
    print()
    print(f"source - random   = {src - rnd:+.4f}   (how much the source order concentrates similarity)")
    print(f"shuffled - random = {shuf - rnd:+.4f}  (should be ~0 if the permutation worked)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(
            {"parquet": str(args.parquet), "molecules": n, "results": rows}, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
