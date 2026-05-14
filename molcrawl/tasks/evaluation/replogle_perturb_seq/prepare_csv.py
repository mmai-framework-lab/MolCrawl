"""Materialise the Replogle Perturb-seq evaluator CSV from a public mirror.

The CellxGene H5AD that the original eval-data-replogle-perturb-seq.sh
points at returns 403 anonymously. The TruthSeq figshare release
(figshare 31840141) ships the same Replogle 2022 K562 atlas in a
single ~150 MB long-format parquet:

    knocked_down_gene | affected_gene | z_score | cell_line

(7,639 KO targets × 8,246 affected genes, 37.7 M non-NaN pairs.)

This module:

1. Downloads the parquet (idempotent — skips if already on disk).
2. Pivots to wide format (rows=perturbations, cols=genes, value=z-score).
3. Optionally subsamples the top-N perturbations and top-K
   highest-variance genes (keeps the evaluator CSV small enough for
   smoke runs).
4. Emits the CSV the existing :func:`load_replogle` reader expects:
   ``perturbation,baseline,perturbed`` with ``baseline`` = zero vector
   and ``perturbed`` = the z-score vector serialised as JSON. The
   evaluator computes ``delta = perturbed - baseline`` so this matches
   the LFC / z-score semantics one-to-one.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_PARQUET_URL = "https://ndownloader.figshare.com/files/63037363"
_DEFAULT_PARQUET_NAME = "replogle_knockdown_effects.parquet"
_ENSEMBL_BATCH_URL = "https://rest.ensembl.org/lookup/symbol/homo_sapiens"


def _build_or_load_symbol_mapping(
    symbols: List[str], cache_path: Path, batch_size: int = 500
) -> dict:
    """Resolve HGNC symbols to Ensembl gene IDs, with on-disk caching.

    The cache is a CSV with two columns ``symbol,ensg``. Symbols with no
    Ensembl match are stored with an empty ``ensg`` so future runs do
    not re-query them. Calls are batched (POST /lookup/symbol) so 7,000
    symbols resolve in ~5-10 requests rather than 7,000.
    """
    import pandas as pd
    import requests

    cache_path = Path(cache_path)
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path).fillna({"ensg": ""})
        mapping = dict(zip(cache_df["symbol"].astype(str), cache_df["ensg"].astype(str)))
        logger.info("Loaded %d cached symbol mappings from %s", len(mapping), cache_path)
    else:
        mapping = {}

    pending = [s for s in symbols if s not in mapping]
    if not pending:
        return mapping
    logger.info(
        "Resolving %d HGNC symbols via Ensembl REST (batch_size=%d)",
        len(pending),
        batch_size,
    )
    for i in range(0, len(pending), batch_size):
        chunk = pending[i : i + batch_size]
        try:
            r = requests.post(
                _ENSEMBL_BATCH_URL,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={"symbols": chunk},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning(
                "Ensembl batch %d/%d failed (%s); marking chunk unmapped",
                i // batch_size + 1,
                (len(pending) + batch_size - 1) // batch_size,
                exc,
            )
            data = {}
        for s in chunk:
            mapping[s] = str(data.get(s, {}).get("id", "") or "")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"symbol": k, "ensg": v} for k, v in sorted(mapping.items())]
    ).to_csv(cache_path, index=False)
    n_resolved = sum(1 for v in mapping.values() if v)
    logger.info(
        "Cached %d symbol mappings (%d resolved, %d unmapped) -> %s",
        len(mapping),
        n_resolved,
        len(mapping) - n_resolved,
        cache_path,
    )
    return mapping


def _download_parquet(parquet_path: Path, url: str) -> None:
    """Idempotent download via curl."""
    import subprocess

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists() and parquet_path.stat().st_size > 1_000_000:
        logger.info("Parquet already present (%d bytes); skipping download", parquet_path.stat().st_size)
        return
    logger.info("Downloading %s -> %s", url, parquet_path)
    subprocess.check_call(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--retry-delay",
            "5",
            "-o",
            str(parquet_path) + ".part",
            url,
        ]
    )
    Path(str(parquet_path) + ".part").rename(parquet_path)


def materialise_replogle_csv(
    output_csv: Path,
    parquet_path: Optional[Path] = None,
    parquet_url: str = _DEFAULT_PARQUET_URL,
    cell_line: str = "K562",
    max_perturbations: Optional[int] = None,
    max_genes: Optional[int] = None,
    symbol_to_ensg_cache: Optional[Path] = None,
    seed: int = 42,
) -> dict:
    import pandas as pd

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if parquet_path is None:
        parquet_path = output_csv.parent / _DEFAULT_PARQUET_NAME
    else:
        parquet_path = Path(parquet_path)
    _download_parquet(parquet_path, parquet_url)

    logger.info("Loading parquet ...")
    df_long = pd.read_parquet(
        parquet_path, columns=["knocked_down_gene", "affected_gene", "z_score", "cell_line"]
    )
    if cell_line:
        before = len(df_long)
        df_long = df_long[df_long["cell_line"] == cell_line]
        logger.info(
            "filtered cell_line=%r: %d -> %d rows", cell_line, before, len(df_long)
        )
        if df_long.empty:
            available = pd.read_parquet(parquet_path, columns=["cell_line"])["cell_line"].unique()
            raise ValueError(
                f"No rows for cell_line={cell_line!r}. "
                f"Available: {sorted(available)}"
            )

    logger.info("Pivoting to wide format (perturbation x gene)...")
    wide = (
        df_long.pivot_table(
            index="knocked_down_gene",
            columns="affected_gene",
            values="z_score",
            aggfunc="mean",
        )
        .fillna(0.0)
        .sort_index()
    )
    logger.info("Wide matrix: %d perturbations x %d genes", *wide.shape)

    if max_genes is not None and max_genes < wide.shape[1]:
        # Pick the highest-variance genes — they carry the most signal.
        gene_var = wide.var(axis=0)
        top_genes = gene_var.sort_values(ascending=False).head(int(max_genes)).index
        wide = wide[top_genes]
        logger.info(
            "Restricted to top-%d highest-variance genes (out of %d)",
            int(max_genes),
            len(gene_var),
        )

    if max_perturbations is not None and max_perturbations < wide.shape[0]:
        # Pick the perturbations with the largest mean |z| — the strongest KO effects.
        per_pert_strength = wide.abs().mean(axis=1)
        top_perts = (
            per_pert_strength.sort_values(ascending=False).head(int(max_perturbations)).index
        )
        wide = wide.loc[top_perts]
        logger.info(
            "Restricted to top-%d strongest perturbations (out of %d)",
            int(max_perturbations),
            len(per_pert_strength),
        )

    n_genes = wide.shape[1]
    zero_vec = json.dumps([0.0] * n_genes)

    # Optional HGNC symbol -> ENSG ID mapping. The molcrawl rna BERT and
    # rnaformer vocabs are keyed by ENSG IDs; without this translation
    # the encoder treats every perturbation as [UNK].
    if symbol_to_ensg_cache is not None:
        cache_path = Path(symbol_to_ensg_cache)
        mapping = _build_or_load_symbol_mapping(
            symbols=[str(s) for s in wide.index],
            cache_path=cache_path,
        )
        unmapped = [s for s in wide.index if not mapping.get(str(s))]
        if unmapped:
            logger.warning(
                "Dropping %d perturbations with no Ensembl mapping (e.g. %s)",
                len(unmapped),
                unmapped[:5],
            )
            wide = wide.loc[[s for s in wide.index if mapping.get(str(s))]]
    else:
        mapping = None

    rows: List[dict] = []
    for pert in wide.index:
        if mapping is not None:
            label = mapping.get(str(pert)) or str(pert)
        else:
            label = str(pert)
        rows.append(
            {
                "perturbation": label,
                "baseline": zero_vec,
                "perturbed": json.dumps(wide.loc[pert].astype(float).tolist()),
            }
        )

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    summary = {
        "output_csv": str(output_csv),
        "parquet_path": str(parquet_path),
        "cell_line": cell_line,
        "n_perturbations": int(wide.shape[0]),
        "n_genes": int(wide.shape[1]),
        "seed": seed,
    }
    logger.info("Wrote %s", summary)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Materialise the Replogle Perturb-seq evaluator CSV from TruthSeq's figshare release"
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--parquet-path",
        default=None,
        help="Where to cache the source parquet (default: alongside output CSV).",
    )
    parser.add_argument("--parquet-url", default=_DEFAULT_PARQUET_URL)
    parser.add_argument(
        "--cell-line",
        default="K562",
        help="Filter rows to this cell line (default: K562 — the only one in TruthSeq).",
    )
    parser.add_argument(
        "--max-perturbations",
        type=int,
        default=None,
        help="Restrict to the top-N strongest perturbations (mean |z|). "
        "Default: no cap.",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Restrict to the top-K highest-variance genes. Default: no cap.",
    )
    parser.add_argument(
        "--symbol-to-ensg-cache",
        default=None,
        help="If set, resolve HGNC symbols to Ensembl gene IDs via the "
        "Ensembl REST API (batched POST /lookup/symbol/homo_sapiens) "
        "and cache to this CSV path. Required when the downstream "
        "encoder vocab is keyed by ENSG IDs (e.g. molcrawl rna BERT / "
        "rnaformer); perturbations with no ENSG match are dropped.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    materialise_replogle_csv(
        output_csv=Path(args.output_csv),
        parquet_path=Path(args.parquet_path) if args.parquet_path else None,
        parquet_url=args.parquet_url,
        cell_line=args.cell_line,
        max_perturbations=args.max_perturbations,
        max_genes=args.max_genes,
        symbol_to_ensg_cache=Path(args.symbol_to_ensg_cache)
        if args.symbol_to_ensg_cache
        else None,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
