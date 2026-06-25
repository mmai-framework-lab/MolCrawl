"""Materialise the Tabula Sapiens tokenised-cell JSONL.

The CellxGene H5AD originally referenced by the downloader returned
403 anonymously (dataset retracted / URL rotated), but the broader
``Tabula Sapiens`` collection (CellxGene collection
e5f58829-1a66-40b5-a624-9046778e74f5) still serves per-tissue H5AD
slices that ARE reachable anonymously. By default we materialise the
smallest slice (Testis, ~0.39 GB); override TABULA_DATASET_URL to
pick a different organ.

The evaluator expects JSONL with rows of the shape::

    {"tokens": [...int ids...], "cell_type": "...", "tissue": "..."}

For each cell we:

1. Find the top-N highest-expression genes (input format used by the
   molcrawl rna BERT pretraining pipeline).
2. Map gene IDs (``var.feature_id`` / ``var_names``) to the encoder
   tokenizer's vocab — usually keyed by ENSG IDs. Genes outside the
   vocab are dropped per cell.
3. Write the resulting token-id list together with the cell-type
   label and tissue tag.

Optional subsampling caps the JSONL size for smoke runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _download(h5ad_path: Path, url: str) -> None:
    """Idempotent curl download."""
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    if h5ad_path.exists() and h5ad_path.stat().st_size > 100_000_000:
        logger.info(
            "H5AD already present (%d bytes); skipping download",
            h5ad_path.stat().st_size,
        )
        return
    logger.info("Downloading %s -> %s", url, h5ad_path)
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
            str(h5ad_path) + ".part",
            url,
        ]
    )
    Path(str(h5ad_path) + ".part").rename(h5ad_path)


def _load_tokenizer_vocab(tokenizer_dir: Path) -> dict:
    """Return ``{token_str: token_id}`` from a HuggingFace tokenizer dir."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return tok.get_vocab()


def materialise_tabula_jsonl(
    h5ad_url: str,
    h5ad_path: Path,
    output_jsonl: Path,
    tokenizer_dir: Path,
    top_n_genes_per_cell: int = 1024,
    max_cells: Optional[int] = None,
    cell_type_field: str = "cell_type",
    tissue_field: str = "tissue",
    seed: int = 42,
) -> dict:
    import anndata as ad

    h5ad_path = Path(h5ad_path)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    _download(h5ad_path, h5ad_url)

    logger.info("Loading H5AD: %s", h5ad_path)
    adata = ad.read_h5ad(str(h5ad_path))
    logger.info(
        "AnnData shape=%s, obs cols=%s, var cols=%s",
        adata.shape,
        list(adata.obs.columns)[:6],
        list(adata.var.columns)[:6],
    )

    if cell_type_field not in adata.obs.columns:
        raise ValueError(
            f"obs missing column {cell_type_field!r}. Available: {list(adata.obs.columns)}"
        )

    # Resolve gene-symbol/ID column to use for tokenizer lookup. CellxGene
    # H5ADs typically have ``var_names`` = ENSG ID and a HGNC symbol in
    # ``var['feature_name']``. We support ENSG-keyed and symbol-keyed
    # tokenizer vocabs by trying both.
    vocab = _load_tokenizer_vocab(Path(tokenizer_dir))
    var_names = list(adata.var_names.astype(str))
    feature_names = (
        list(adata.var["feature_name"].astype(str))
        if "feature_name" in adata.var.columns
        else var_names
    )

    var_in_vocab_ensg = sum(1 for v in var_names if v in vocab)
    var_in_vocab_sym = sum(1 for v in feature_names if v in vocab)
    logger.info(
        "Vocab overlap: ENSG via var_names=%d/%d  HGNC via feature_name=%d/%d",
        var_in_vocab_ensg,
        len(var_names),
        var_in_vocab_sym,
        len(feature_names),
    )
    use_ensg = var_in_vocab_ensg >= var_in_vocab_sym
    keys = var_names if use_ensg else feature_names
    logger.info("Using %s as tokenizer key", "ENSG IDs" if use_ensg else "HGNC symbols")

    # Per-gene token id (or -1 if unknown)
    import numpy as np

    gene_token_ids = np.array(
        [int(vocab.get(k, -1)) for k in keys], dtype=np.int64
    )
    n_known = int((gene_token_ids >= 0).sum())
    logger.info("Per-gene token ids resolved: %d / %d", n_known, len(gene_token_ids))
    if n_known == 0:
        raise RuntimeError(
            "No gene from the H5AD overlaps the tokenizer vocab; cannot tokenise. "
            "Pass a tokenizer whose vocab keys match the AnnData var column."
        )

    # Subsample cells reproducibly, optionally
    rng = random.Random(seed)
    n_cells = adata.n_obs
    if max_cells is not None and max_cells < n_cells:
        keep = sorted(rng.sample(range(n_cells), int(max_cells)))
        adata = adata[keep, :].copy()
        logger.info("Subsampled to %d cells (seed=%d)", adata.n_obs, seed)

    cell_types = adata.obs[cell_type_field].astype(str).tolist()
    tissues = (
        adata.obs[tissue_field].astype(str).tolist()
        if tissue_field in adata.obs.columns
        else [""] * adata.n_obs
    )

    # Use the .X matrix; for CellxGene the canonical normalised counts
    # live in obs.layers but ``X`` is fine for ranked-by-expression.
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()

    n_written = 0
    skipped = 0
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for i in range(adata.n_obs):
            if sp.issparse(X):
                row = X.getrow(i)
                col_idx = row.indices
                vals = row.data
            else:
                row = np.asarray(X[i]).ravel()
                col_idx = np.where(row != 0)[0]
                vals = row[col_idx]
            if col_idx.size == 0:
                skipped += 1
                continue
            # Sort by expression descending, take top-N, drop unknown genes
            order = np.argsort(-vals)
            top = col_idx[order]
            tok_ids: List[int] = []
            for c in top:
                tid = int(gene_token_ids[int(c)])
                if tid < 0:
                    continue
                tok_ids.append(tid)
                if len(tok_ids) >= top_n_genes_per_cell:
                    break
            if not tok_ids:
                skipped += 1
                continue
            rec = {
                "tokens": tok_ids,
                "cell_type": cell_types[i],
                "tissue": tissues[i],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    summary = {
        "h5ad": str(h5ad_path),
        "output_jsonl": str(output_jsonl),
        "n_cells_written": n_written,
        "n_cells_skipped_empty": skipped,
        "tokenizer": str(tokenizer_dir),
        "tokenizer_key": "ENSG" if use_ensg else "HGNC",
        "n_known_genes": n_known,
        "n_total_genes": len(gene_token_ids),
        "top_n_genes_per_cell": top_n_genes_per_cell,
    }
    logger.info("Wrote %s", summary)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Materialise the Tabula Sapiens tokenised-cell JSONL"
    )
    parser.add_argument(
        "--h5ad-url",
        default="https://datasets.cellxgene.cziscience.com/abec77b5-d7b2-4a83-8111-27f4dc8614dd.h5ad",
        help="CellxGene H5AD asset URL (default: Tabula Sapiens — Testis, 0.39 GB).",
    )
    parser.add_argument(
        "--h5ad-path",
        required=True,
        help="Local cache path for the downloaded H5AD.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        help="HF tokenizer directory whose vocab is keyed by ENSG IDs "
        "or HGNC symbols (typically "
        "$LEARNING_SOURCE_DIR/rna/custom_tokenizer_bert).",
    )
    parser.add_argument("--top-n-genes-per-cell", type=int, default=1024)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--cell-type-field", default="cell_type")
    parser.add_argument("--tissue-field", default="tissue")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    materialise_tabula_jsonl(
        h5ad_url=args.h5ad_url,
        h5ad_path=Path(args.h5ad_path),
        output_jsonl=Path(args.output_jsonl),
        tokenizer_dir=Path(args.tokenizer_dir),
        top_n_genes_per_cell=args.top_n_genes_per_cell,
        max_cells=args.max_cells,
        cell_type_field=args.cell_type_field,
        tissue_field=args.tissue_field,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
