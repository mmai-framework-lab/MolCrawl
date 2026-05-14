"""Reshape the DeepLoc 2.0 multi-label CSV into the evaluator's schema.

The upstream CSV ships 10 binary class columns plus metadata:

    ACC, Kingdom, Partition, Membrane,
    Cytoplasm, Nucleus, Extracellular, Cell membrane, Mitochondrion,
    Plastid, Endoplasmic reticulum, Lysosome/Vacuole, Golgi apparatus,
    Peroxisome, Sequence

The :class:`DeepLocEvaluator` consumes a single-label format with:

    sequence, localisation, cluster_id, kingdom

This module emits the latter, picking the *dominant* localisation per
protein. Ties are broken by class-prevalence order (most-frequent class
in the corpus wins); proteins with no positive label are dropped. The
upstream ``Partition`` column is renamed to ``cluster_id`` so
:func:`splits.cluster_split` can consume it directly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# Order matches molcrawl.tasks.evaluation.deeploc.data_preparation.DEEPLOC_CLASSES.
_CLASSES = (
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
)


def reshape_deeploc_csv(
    source_csv: Path,
    output_csv: Path,
) -> dict:
    """Read the multi-label CSV and emit a single-label version."""
    import pandas as pd

    source_csv = Path(source_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)
    logger.info("Loaded %d rows from %s (cols=%s)", len(df), source_csv, list(df.columns))

    missing = [c for c in _CLASSES if c not in df.columns]
    if missing:
        raise ValueError(
            f"DeepLoc source missing class columns: {missing}. "
            f"Got: {list(df.columns)}"
        )
    if "Sequence" not in df.columns:
        raise ValueError("DeepLoc source missing 'Sequence' column")

    label_block = df[list(_CLASSES)].fillna(0).astype(float)
    # Class-prevalence order, used to break argmax ties deterministically.
    prevalence = label_block.sum().sort_values(ascending=False).index.tolist()
    rank = {cls: i for i, cls in enumerate(prevalence)}

    def _dominant(row) -> Optional[str]:
        positives = [cls for cls in _CLASSES if row[cls] >= 1.0]
        if not positives:
            return None
        # Lower rank = more prevalent class; pick that one.
        positives.sort(key=lambda c: rank[c])
        return positives[0]

    df["localisation"] = label_block.apply(_dominant, axis=1)

    n_before = len(df)
    df = df.dropna(subset=["localisation", "Sequence"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("Dropped %d rows with no positive label or missing sequence", n_dropped)

    out_cols = {
        "sequence": df["Sequence"].astype(str),
        "localisation": df["localisation"].astype(str),
    }
    if "Partition" in df.columns:
        out_cols["cluster_id"] = df["Partition"].astype(int)
    if "Kingdom" in df.columns:
        out_cols["kingdom"] = df["Kingdom"].astype(str)
    if "ACC" in df.columns:
        out_cols["accession"] = df["ACC"].astype(str)

    out = (
        __import__("pandas")
        .DataFrame(out_cols)
        .reset_index(drop=True)
    )
    out.to_csv(output_csv, index=False)

    summary = {
        "output": str(output_csv),
        "n_rows": int(len(out)),
        "n_dropped_unlabeled": int(n_dropped),
        "class_counts": out["localisation"].value_counts().to_dict(),
    }
    logger.info(
        "Wrote %d single-label rows -> %s. Class counts: %s",
        summary["n_rows"],
        output_csv,
        summary["class_counts"],
    )
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Reshape DeepLoc 2.0 multi-label CSV into single-label evaluator format"
    )
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args(argv)

    reshape_deeploc_csv(
        source_csv=Path(args.source_csv),
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
