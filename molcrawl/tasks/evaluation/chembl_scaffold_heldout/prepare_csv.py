"""Build the scaffold-disjoint train/heldout CSVs for ChEMBL.

The training pipeline emits a flat ``smiles.txt`` (≈ 2.7 M canonical
SMILES) under ``$LEARNING_SOURCE_DIR/compounds/chembl/chembl_db/``.
For a held-out *likelihood* benchmark we need a test set whose
Bemis-Murcko scaffolds are disjoint from the training scaffolds —
otherwise the SMILES has effectively been seen and the perplexity
number is over-optimistic.

What this module does:

1. Read SMILES from ``smiles.txt`` (or any one-SMILES-per-line file),
   optionally subsampling to a manageable scale via reservoir
   sampling (keeps memory bounded for the 2.7 M-row source file).
2. Compute the Bemis-Murcko scaffold for each SMILES (RDKit).
3. Sort scaffolds by frequency: large/common scaffolds → train,
   rare scaffolds → heldout. This mirrors the MoleculeNet
   convention and ensures the held-out set is genuinely OOD.
4. Write ``train.csv`` and ``heldout.csv`` with a single ``smiles``
   column. Both CSVs are deterministic given ``seed``.

The heldout split aims for ``heldout_frac`` of total sampled rows;
exact size depends on scaffold cluster sizes (we take whole
scaffolds, never split a scaffold across train/heldout).
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _bemis_murcko_scaffold(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "rdkit is required for scaffold computation; "
            "install via `mamba install -c conda-forge rdkit`."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or ""


def _read_smiles(
    path: Path,
    max_source: Optional[int] = None,
    seed: int = 42,
) -> List[str]:
    """Stream SMILES from ``path``; optional reservoir-sample to ``max_source``."""
    rng = random.Random(seed)
    out: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        if max_source is None:
            for raw in fh:
                smi = raw.strip()
                if smi:
                    out.append(smi)
            logger.info("Read %d SMILES from %s (no subsample)", len(out), path)
            return out
        # Vitter reservoir sampling — bounded memory for huge inputs.
        for i, raw in enumerate(fh):
            smi = raw.strip()
            if not smi:
                continue
            if len(out) < max_source:
                out.append(smi)
            else:
                j = rng.randint(0, i)
                if j < max_source:
                    out[j] = smi
    logger.info(
        "Read SMILES with reservoir subsample: kept %d of streamed input from %s",
        len(out),
        path,
    )
    return out


def _build_scaffold_groups(
    smiles_list: Iterable[str],
) -> Tuple[List[List[int]], List[int]]:
    """Group indices by scaffold; returns (sorted_buckets, dropped_indices).

    Returned ``buckets`` are sorted by descending size, then by first-seen
    index so the result is deterministic. ``dropped`` lists indices the
    SMILES of which RDKit failed to parse (those are NOT included in the
    buckets).
    """
    scaffolds: dict = defaultdict(list)
    dropped: List[int] = []
    for idx, smi in enumerate(smiles_list):
        sc = _bemis_murcko_scaffold(smi)
        if sc is None:
            dropped.append(idx)
            continue
        scaffolds[sc].append(idx)
    buckets = sorted(scaffolds.values(), key=lambda b: (-len(b), b[0]))
    logger.info(
        "Scaffold grouping: %d unique scaffolds across %d parseable SMILES "
        "(%d unparseable)",
        len(buckets),
        sum(len(b) for b in buckets),
        len(dropped),
    )
    return buckets, dropped


def _split_scaffolds_disjoint(
    buckets: List[List[int]],
    heldout_frac: float,
) -> Tuple[List[int], List[int]]:
    """Take rare scaffolds first into heldout until quota is met.

    We iterate the buckets in *ascending* size (singletons first) so the
    heldout set is composed of structurally unusual molecules — the
    hardest possible perplexity test. Anything left goes to train.
    """
    n_total = sum(len(b) for b in buckets)
    target_heldout = int(round(n_total * heldout_frac))

    # Iterate from smallest scaffold cluster up.
    ascending = sorted(buckets, key=lambda b: (len(b), b[0]))

    heldout: List[int] = []
    train_buckets: List[List[int]] = []
    for bucket in ascending:
        if len(heldout) + len(bucket) <= target_heldout:
            heldout.extend(bucket)
        else:
            train_buckets.append(bucket)
    train: List[int] = [i for b in train_buckets for i in b]

    logger.info(
        "Scaffold-disjoint split: train=%d heldout=%d (target heldout=%d, "
        "frac=%.3f)",
        len(train),
        len(heldout),
        target_heldout,
        heldout_frac,
    )
    return sorted(train), sorted(heldout)


def prepare_chembl_scaffold_csvs(
    source_smiles: Path,
    output_dir: Path,
    heldout_frac: float = 0.05,
    max_source: Optional[int] = None,
    max_train: Optional[int] = None,
    max_heldout: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """End-to-end driver: smiles.txt → train.csv + heldout.csv.

    Returns a dict of artefact paths + summary counts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles = _read_smiles(Path(source_smiles), max_source=max_source, seed=seed)
    buckets, dropped = _build_scaffold_groups(smiles)
    train_idx, heldout_idx = _split_scaffolds_disjoint(buckets, heldout_frac)

    rng = random.Random(seed)
    if max_train is not None and len(train_idx) > max_train:
        train_idx = rng.sample(train_idx, max_train)
        train_idx.sort()
    if max_heldout is not None and len(heldout_idx) > max_heldout:
        heldout_idx = rng.sample(heldout_idx, max_heldout)
        heldout_idx.sort()

    train_smiles = [smiles[i] for i in train_idx]
    heldout_smiles = [smiles[i] for i in heldout_idx]

    train_path = output_dir / "train.csv"
    heldout_path = output_dir / "heldout.csv"
    pd.DataFrame({"smiles": train_smiles}).to_csv(train_path, index=False)
    pd.DataFrame({"smiles": heldout_smiles}).to_csv(heldout_path, index=False)

    summary = {
        "train_csv": str(train_path),
        "heldout_csv": str(heldout_path),
        "n_source_total": len(smiles),
        "n_unparseable": len(dropped),
        "n_train": len(train_smiles),
        "n_heldout": len(heldout_smiles),
        "heldout_frac": heldout_frac,
        "seed": seed,
    }
    summary_path = output_dir / "prepare_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")
    summary["summary_txt"] = str(summary_path)
    logger.info("Wrote ChEMBL scaffold split: %s", summary)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Prepare scaffold-disjoint train/heldout CSVs for ChEMBL"
    )
    parser.add_argument(
        "--source-smiles",
        required=True,
        help="One-SMILES-per-line text file (e.g. learning_source_*/compounds/chembl/chembl_db/smiles.txt)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout-frac", type=float, default=0.05)
    parser.add_argument(
        "--max-source",
        type=int,
        default=None,
        help="Reservoir-subsample source SMILES to this many rows before scaffold "
        "computation (e.g. 200000 keeps RDKit's per-molecule scaffold pass tractable).",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Cap the train.csv row count (random subsample after splitting).",
    )
    parser.add_argument(
        "--max-heldout",
        type=int,
        default=None,
        help="Cap the heldout.csv row count (random subsample after splitting).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    prepare_chembl_scaffold_csvs(
        source_smiles=Path(args.source_smiles),
        output_dir=Path(args.output_dir),
        heldout_frac=args.heldout_frac,
        max_source=args.max_source,
        max_train=args.max_train,
        max_heldout=args.max_heldout,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
