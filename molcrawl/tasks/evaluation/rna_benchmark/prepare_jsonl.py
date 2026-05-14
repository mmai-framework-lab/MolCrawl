"""Build the per-tissue JSONL the rna_benchmark evaluator consumes.

The training pipeline emits per-tissue parquet shards under
``$LEARNING_SOURCE_DIR/rna/parquet_files/<tissue>.<chunk>.parquet``,
each with two columns:

- ``token`` — list of int16 token ids in the rna BERT / rnaformer vocab
- ``token_count`` — original cell length

For evaluation we want a small, balanced JSONL where each record is

    {"dataset": <tissue>, "tokens": [...], "token_count": N}

The rna_benchmark evaluator groups by ``dataset`` and reports per-group
mean log-likelihood / perplexity. The HfMlm adapter has been extended
to accept pre-tokenised int lists so we ship the raw ids through; no
tokenizer round-trip is required.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _list_tissues(parquet_dir: Path) -> dict[str, List[Path]]:
    """Group parquet files by tissue (filename prefix before the first dot)."""
    out: dict[str, List[Path]] = {}
    for path in sorted(parquet_dir.glob("*.parquet")):
        tissue = path.name.split(".", 1)[0]
        out.setdefault(tissue, []).append(path)
    return out


def _sample_from_parquet(
    files: Iterable[Path],
    n_target: int,
    seed: int,
) -> List[dict]:
    """Reservoir-sample ``n_target`` cells across the given parquet files.

    Each record returned is ``{"tokens": [...], "token_count": N}`` and is
    suitable for direct JSONL serialisation.
    """
    import pandas as pd

    rng = random.Random(seed)
    reservoir: List[dict] = []
    seen = 0
    for path in files:
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            cell = {
                "tokens": [int(t) for t in row["token"]],
                "token_count": int(row.get("token_count", len(row["token"]))),
            }
            if len(reservoir) < n_target:
                reservoir.append(cell)
            else:
                j = rng.randint(0, seen)
                if j < n_target:
                    reservoir[j] = cell
            seen += 1
            if n_target <= 0:
                break
        if seen >= 50_000 and n_target * 5 < seen:
            # Plenty of pool covered; stop reading more shards once we're
            # well past the target × 5 mark for a tissue.
            break
    return reservoir


def prepare_rna_benchmark_jsonl(
    parquet_dir: Path,
    output_path: Path,
    cells_per_tissue: int = 100,
    max_tissues: Optional[int] = None,
    tissues: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> dict:
    parquet_dir = Path(parquet_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tissue_files = _list_tissues(parquet_dir)
    if tissues is not None:
        wanted = {t.strip() for t in tissues}
        tissue_files = {k: v for k, v in tissue_files.items() if k in wanted}
    if max_tissues is not None:
        tissue_files = dict(list(tissue_files.items())[: int(max_tissues)])

    summary: Dict[str, Any] = {"output": str(output_path), "tissues": {}, "total_cells": 0}
    with output_path.open("w", encoding="utf-8") as fh:
        for tissue, files in tissue_files.items():
            logger.info(
                "Sampling %d cells from %d parquet shards for tissue=%r",
                cells_per_tissue,
                len(files),
                tissue,
            )
            cells = _sample_from_parquet(
                files, n_target=cells_per_tissue, seed=seed
            )
            for c in cells:
                fh.write(
                    json.dumps(
                        {
                            "dataset": tissue,
                            "tokens": c["tokens"],
                            "token_count": c["token_count"],
                        }
                    )
                    + "\n"
                )
            summary["tissues"][tissue] = len(cells)
            summary["total_cells"] += len(cells)

    logger.info(
        "Wrote %d cells across %d tissues -> %s",
        summary["total_cells"],
        len(summary["tissues"]),
        output_path,
    )
    summary_path = output_path.parent / (output_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Build the rna_benchmark JSONL from training-pipeline parquet shards"
    )
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Directory containing <tissue>.<chunk>.parquet files "
        "(typically $LEARNING_SOURCE_DIR/rna/parquet_files/).",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cells-per-tissue", type=int, default=100)
    parser.add_argument(
        "--max-tissues",
        type=int,
        default=None,
        help="If set, keep only the first N tissues (alphabetical).",
    )
    parser.add_argument(
        "--tissues",
        nargs="*",
        default=None,
        help="If set, restrict to this list of tissue names "
        "(matched against the parquet filename prefix).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    prepare_rna_benchmark_jsonl(
        parquet_dir=Path(args.parquet_dir),
        output_path=Path(args.output_path),
        cells_per_tissue=args.cells_per_tissue,
        max_tissues=args.max_tissues,
        tissues=args.tissues,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
