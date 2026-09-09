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
import hashlib
import json
import logging
import pickle
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


def _gene_median_path(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    from molcrawl.data.rna.dataset.geneformer.tokenizer import GENE_MEDIAN_FILE

    return Path(GENE_MEDIAN_FILE)


def _sha256(path: Path) -> str:
    """So a report can say which medians produced a given JSONL."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_gene_medians(explicit: Optional[Path] = None) -> dict:
    """The medians pretraining used -- never recomputed from evaluation data."""
    path = _gene_median_path(explicit)
    if not path.exists():
        raise FileNotFoundError(f"gene median dictionary not found: {path}")
    with path.open("rb") as fh:
        medians = pickle.load(fh)
    logger.info("Gene medians: %d genes from %s", len(medians), path)
    return medians


def materialise_tabula_jsonl(
    h5ad_url: str,
    h5ad_path: Path,
    output_jsonl: Path,
    tokenizer_dir: Path,
    top_n_genes_per_cell: int = 1024,
    max_cells: Optional[int] = None,
    cell_type_field: str = "cell_type",
    tissue_field: str = "tissue",
    cell_id_field: str = "soma_joinid",
    gene_median_file: Optional[Path] = None,
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
    # Downstream analysis joins the model's embeddings back to a count matrix
    # (scIB, the HVG and scVI controls). The join has to be on an identifier
    # carried per row: rows are skipped when a cell has no gene in the vocab, so
    # line number and cell index are not the same sequence.
    cell_ids_from_obs_column = bool(cell_id_field) and cell_id_field in adata.obs.columns
    if cell_ids_from_obs_column:
        cell_ids = adata.obs[cell_id_field].astype(str).tolist()
    else:
        cell_ids = adata.obs_names.astype(str).tolist()
        logger.warning(
            "No %s column in obs; falling back to obs_names for cell_id. "
            "obs_names are assigned per study, so they do not identify a cell "
            "across datasets.",
            cell_id_field,
        )
    tissues = (
        adata.obs[tissue_field].astype(str).tolist()
        if tissue_field in adata.obs.columns
        else [""] * adata.n_obs
    )

    # Rank on the value pretraining ranked on, not on the raw counts.
    #
    # Geneformer's rank value encoding orders genes by
    #     X / n_counts * 10_000 / (that gene's non-zero median in Genecorpus-30M)
    # (data/rna/dataset/geneformer/tokenizer.py:200). Sorting the raw counts
    # instead leaves the top 1,024 about 87% the same, so roughly one gene in
    # eight is a gene the model never saw at that rank -- and the ordering *is*
    # the input for this encoding. Measured 2026-09-08 on 5,000 lung-atlas cells.
    #
    # n_counts is the corpus-side ``raw_sum``: the total over *all* genes
    # (cellxgene/script/h5ad_to_loom.py:31), not over the median-dictionary
    # subset the numerator is restricted to. Using the subset sum here would be
    # a different per-cell constant -- harmless for the order, but no longer the
    # same expression, so take raw_sum when the obs carries it.
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()

    medians = _load_gene_medians(gene_median_file)
    # The training gene set is exactly the median dictionary's keys
    # (tokenizer.py:95 -> genelist_dict), so a gene without a median was never
    # tokenised and must not be ranked here either.
    median_vec = np.array([medians.get(k, np.nan) for k in keys], dtype=float)
    rankable = np.isfinite(median_vec) & (gene_token_ids >= 0)
    logger.info(
        "Rankable genes: %d / %d (in median dictionary and in vocab)",
        int(rankable.sum()), len(keys),
    )
    if not rankable.any():
        raise RuntimeError(
            "No gene of this H5AD has both a token id and a non-zero median; "
            "cannot reproduce the pretraining ranking."
        )

    if "raw_sum" in adata.obs.columns:
        n_counts = adata.obs["raw_sum"].to_numpy(dtype=float)
        n_counts_source = "obs.raw_sum"
    else:
        n_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)
        n_counts_source = "row sum over all genes (obs.raw_sum absent)"
        logger.warning(
            "obs has no raw_sum; using the row sum over all genes. This matches "
            "raw_sum when X holds the full raw counts, and does not otherwise."
        )
    n_counts[n_counts <= 0] = 1.0
    logger.info("n_counts from %s", n_counts_source)

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
            # Same expression the tokenizer used, then descending order.
            keep = rankable[col_idx]
            cols_k, vals_k = col_idx[keep], vals[keep]
            if cols_k.size == 0:
                skipped += 1
                continue
            scaled = vals_k / n_counts[i] * 10_000.0 / median_vec[cols_k]
            order = np.argsort(-scaled)[:top_n_genes_per_cell]
            tok_ids: List[int] = [int(t) for t in gene_token_ids[cols_k[order]]]
            if not tok_ids:
                skipped += 1
                continue
            rec = {
                "cell_id": cell_ids[i],
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
        "cell_id_field": cell_id_field if cell_ids_from_obs_column else "obs_names",
        "rank_basis": "geneformer median-scaled (X / n_counts * 10000 / gene median)",
        "n_counts_source": n_counts_source,
        "gene_median_file": str(_gene_median_path(gene_median_file)),
        "gene_median_sha256": _sha256(_gene_median_path(gene_median_file)),
        "n_rankable_genes": int(rankable.sum()),
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
    parser.add_argument(
        "--cell-id-field",
        default="soma_joinid",
        help="obs column written to each record as cell_id, for joining the "
        "embeddings back to a count matrix. Falls back to obs_names when the "
        "column is absent.",
    )
    parser.add_argument(
        "--gene-median-file",
        default=None,
        help="Pickle of per-gene non-zero medians. Defaults to the file "
        "pretraining used; pass one only to reproduce an older run.",
    )
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
        cell_id_field=args.cell_id_field,
        gene_median_file=Path(args.gene_median_file) if args.gene_median_file else None,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
