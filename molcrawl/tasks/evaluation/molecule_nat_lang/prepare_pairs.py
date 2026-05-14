"""Build the (molecule, caption) pair CSV the molecule_nat_lang evaluator consumes.

Source: Mol-Instructions ``molecular_description_generation`` HF dataset, which
ships under
``$LEARNING_SOURCE_DIR/molecule_nat_lang/mol_instructions/zjunlp_Mol-Instructions/``
after the training prep step. Schema:

- ``input``        — SELFIES (or SMILES) string
- ``output``       — natural-language caption
- ``instruction``  — task prompt (constant for this subset)
- ``metadata``     — JSON-serialised stats (split, task)

For evaluation we want a CSV with ``smiles`` + ``caption`` columns.
The molecule_nat_lang model was trained on the same encoding, so the
``input`` field flows through as-is (no SELFIES → SMILES conversion).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def prepare_pairs_csv(
    source_dir: Path,
    output_csv: Path,
    split: str = "test",
    max_pairs: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Read the Mol-Instructions HF dataset and emit a flat CSV.

    ``split`` is read from the ``metadata.split`` field (Mol-Instructions
    folds train/valid/test inside one HF Dataset shard rather than three
    separate shards). When the field is missing we fall back to all rows.
    """
    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets is required; install via `pip install datasets`."
        ) from exc
    import pandas as pd

    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(str(source_dir))
    logger.info("Loaded %d rows from %s", len(ds), source_dir)

    df = ds.to_pandas()
    if "metadata" in df.columns and split:
        wanted = split.strip().lower()

        def _split_of(meta: object) -> str:
            if isinstance(meta, dict):
                return str(meta.get("split", "")).lower()
            try:
                return str(json.loads(str(meta).replace("'", '"')).get("split", "")).lower()
            except Exception:
                return ""

        df["_split"] = df["metadata"].map(_split_of)
        kept = df[df["_split"] == wanted]
        if len(kept) == 0:
            logger.warning(
                "No rows for split=%r in metadata; falling back to all rows", wanted
            )
        else:
            df = kept
        df = df.drop(columns=["_split"], errors="ignore")

    df = df.rename(columns={"input": "smiles", "output": "caption"})[
        ["smiles", "caption"]
    ].dropna().reset_index(drop=True)
    logger.info("Filtered to %d rows after split + dropna", len(df))

    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(n=int(max_pairs), random_state=seed).reset_index(drop=True)
        logger.info("Random subsample to %d pairs (seed=%d)", len(df), seed)

    df.to_csv(output_csv, index=False)
    summary = {
        "output": str(output_csv),
        "n_pairs": int(len(df)),
        "split": split,
        "seed": seed,
    }
    logger.info("Wrote %d pairs -> %s", summary["n_pairs"], output_csv)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Build the molecule_nat_lang pair CSV from the Mol-Instructions HF dataset"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="HF dataset directory, typically "
        "$LEARNING_SOURCE_DIR/molecule_nat_lang/mol_instructions/zjunlp_Mol-Instructions/molecular_description_generation",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    prepare_pairs_csv(
        source_dir=Path(args.source_dir),
        output_csv=Path(args.output_csv),
        split=args.split,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
