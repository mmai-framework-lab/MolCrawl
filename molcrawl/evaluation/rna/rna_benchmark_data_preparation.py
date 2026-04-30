#!/usr/bin/env python3
"""
RNA Benchmark Data Preparation Script

scRNA-seq (.h5ad) can be evaluated with BERT/GPT-2
Generate token column data (JSONL).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from molcrawl.rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer
from molcrawl.core.utils.evaluation_output import setup_evaluation_logging


@dataclass
class DatasetConfig:
    """Dataset settings"""

    name: str
    path: Path


def _discover_h5ad_files(benchmark_dir: Path) -> Dict[str, Path]:
    """Enumerate .h5ad in benchmark directory"""
    h5ad_files: Dict[str, Path] = {}
    for file_path in benchmark_dir.glob("*.h5ad"):
        h5ad_files[file_path.stem] = file_path
    return h5ad_files


def _select_datasets(all_files: Dict[str, Path], selected: Optional[List[str]]) -> List[DatasetConfig]:
    """Select the target dataset"""
    if not selected:
        return [DatasetConfig(name=k, path=v) for k, v in sorted(all_files.items())]

    selected_set = {name.strip() for name in selected}
    configs: List[DatasetConfig] = []
    for name in selected_set:
        if name not in all_files:
            raise FileNotFoundError(f"The specified dataset was not found: {name}")
        configs.append(DatasetConfig(name=name, path=all_files[name]))
    return configs


def _resolve_gene_id_column(adata: ad.AnnData, preferred: Optional[str]) -> Tuple[str, List[str]]:
    """Identify gene ID column"""
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["ensembl_id", "gene_id", "gene_ids", "feature_id", "gene"])
    candidates.extend(list(adata.var.columns))
    for col in candidates:
        if col in adata.var.columns:
            return col, list(adata.var[col].astype(str))
    # Fallback: var_names
    return "__var_names__", list(adata.var_names.astype(str))


def _auto_detect_mapping_columns(
    df: pd.DataFrame,
    symbol_column: Optional[str],
    ensembl_column: Optional[str],
) -> Tuple[str, str]:
    """Automatically detect column names in mapping file"""
    symbol_candidates = [col for col in [symbol_column, "hgnc_symbol", "symbol", "gene_symbol"] if col]
    ensembl_candidates = [col for col in [ensembl_column, "ensembl_gene_id", "ensembl_id", "gene_id"] if col]

    resolved_symbol = next((col for col in symbol_candidates if col in df.columns), None)
    resolved_ensembl = next((col for col in ensembl_candidates if col in df.columns), None)

    if resolved_symbol is None or resolved_ensembl is None:
        raise ValueError(
            f"Failed to automatically detect column names in the mapping file. Available columns: {list(df.columns)}"
        )

    return resolved_symbol, resolved_ensembl


def _load_symbol_to_ensembl_map(
    mapping_path: Path,
    symbol_column: Optional[str],
    ensembl_column: Optional[str],
) -> Dict[str, str]:
    """Load gene symbol → Ensembl ID mapping"""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    if mapping_path.suffix.lower() in {".csv"}:
        df = pd.read_csv(mapping_path, comment="#")
    else:
        df = pd.read_csv(mapping_path, sep="\t", comment="#")

    resolved_symbol, resolved_ensembl = _auto_detect_mapping_columns(df, symbol_column, ensembl_column)
    mapping = dict(zip(df[resolved_symbol].astype(str), df[resolved_ensembl].astype(str)))
    return mapping


def _validate_gene_ids(
    gene_ids: List[str],
    known_gene_ids: Dict[str, int],
    dataset_name: str,
    logger,
) -> None:
    """Verify gene ID match rate"""
    if not gene_ids:
        raise ValueError(f"{dataset_name}: Gene ID is empty.")

    match_count = sum(1 for gene_id in gene_ids if gene_id in known_gene_ids)
    match_ratio = match_count / max(len(gene_ids), 1)
    logger.info(f"{dataset_name}: gene_id match ratio = {match_ratio:.2%}")

    if match_ratio < 0.1:
        raise ValueError(f"{dataset_name}: Gene ID match rate too low ({match_ratio:.2%}). Ensembl ID column may be missing.")


def _compute_n_counts(X_view) -> np.ndarray:
    """Calculate the total count per cell from X"""
    if sp.issparse(X_view):
        return np.asarray(X_view.sum(axis=1)).reshape(-1)
    return np.asarray(X_view.sum(axis=1)).reshape(-1)


def _tokenize_anndata_with_gene_ids(
    tokenizer: TranscriptomeTokenizer,
    adata: ad.AnnData,
    gene_ids: List[str],
) -> List[List[int]]:
    """Convert anndata object to token column (gene_id column specification supported)"""
    has_n_counts: bool = "n_counts" in adata.obs.columns

    # Select only coding/miRNA genes
    coding_mirna_mask = np.array([tokenizer.genelist_dict.get(gene_id, False) for gene_id in gene_ids])
    coding_mirna_loc = np.where(coding_mirna_mask)[0]
    if len(coding_mirna_loc) == 0:
        raise ValueError("No valid gene ID was found.")

    norm_factor_vector = np.array([tokenizer.gene_median_dict[gene_ids[i]] for i in coding_mirna_loc])
    coding_mirna_tokens = np.array([tokenizer.gene_token_dict[gene_ids[i]] for i in coding_mirna_loc])

    # If filter_pass is present, limit the target
    if "filter_pass" in adata.obs.columns:
        filter_pass_loc = np.where(adata.obs["filter_pass"].values == 1)[0]
    else:
        filter_pass_loc = np.arange(adata.shape[0])

    tokenized_cells: List[List[int]] = []
    chunk_size: int = 512
    target_sum: float = 10_000

    for i in range(0, len(filter_pass_loc), chunk_size):
        idx = filter_pass_loc[i : i + chunk_size]
        # Backed anndata restricts simultaneous fancy indexing of rows and columns, so
        # Get only the rows first, then extract the columns
        X_rows = adata[idx].X
        if sp.issparse(X_rows):
            X_view = X_rows[:, coding_mirna_loc]
        else:
            X_view = np.asarray(X_rows)[:, coding_mirna_loc]
        if has_n_counts:
            n_counts = adata[idx].obs["n_counts"].values[:, None]
        else:
            # If n_counts is missing, calculate the total count of the target gene and use it instead
            n_counts = _compute_n_counts(X_view)[:, None]
        X_norm = X_view / n_counts * target_sum / norm_factor_vector
        X_norm = sp.csr_matrix(X_norm)

        for row in range(X_norm.shape[0]):
            nonzero = X_norm[row].data
            indices = X_norm[row].indices
            if nonzero.size == 0:
                tokenized_cells.append([])
                continue
            sorted_idx = np.argsort(-nonzero)
            tokenized_cells.append(coding_mirna_tokens[indices][sorted_idx].tolist())

    return tokenized_cells


def _tokenize_h5ad(
    tokenizer: TranscriptomeTokenizer,
    dataset: DatasetConfig,
    max_cells: Optional[int],
    seed: int,
    gene_id_column: Optional[str],
    gene_symbol_map: Optional[Path],
    symbol_column: str,
    ensembl_column: str,
    logger,
) -> List[List[int]]:
    """Read h5ad and generate token string for each cell"""
    random.seed(seed)
    adata = ad.read_h5ad(dataset.path, backed="r")

    gene_id_col, gene_ids = _resolve_gene_id_column(adata, gene_id_column)
    logger.info(f"{dataset.name}: gene_id column = {gene_id_col}")

    # Convert if symbol → Ensembl mapping is specified
    if gene_symbol_map is not None:
        mapping = _load_symbol_to_ensembl_map(gene_symbol_map, symbol_column, ensembl_column)
        converted: List[str] = []
        for gene_id in gene_ids:
            converted.append(mapping.get(gene_id, ""))
        gene_ids = converted
        logger.info(f"{dataset.name}: applied gene symbol mapping ({gene_symbol_map})")

    _validate_gene_ids(gene_ids, tokenizer.gene_token_dict, dataset.name, logger)

    tokenized_cells = _tokenize_anndata_with_gene_ids(tokenizer, adata, gene_ids)

    if max_cells is not None and len(tokenized_cells) > max_cells:
        tokenized_cells = random.sample(tokenized_cells, k=max_cells)

    # Convert numpy array etc. to normal list[int]
    tokens_list: List[List[int]] = []
    for cell_tokens in tokenized_cells:
        if hasattr(cell_tokens, "tolist"):
            tokens_list.append(cell_tokens.tolist())
        else:
            tokens_list.append(list(cell_tokens))
    return tokens_list


def _write_jsonl(
    output_file: Path,
    dataset_name: str,
    tokens_list: Iterable[List[int]],
) -> int:
    """Append and save as JSONL"""
    count = 0
    with output_file.open("a", encoding="utf-8") as f:
        for tokens in tokens_list:
            record = {"dataset": dataset_name, "tokens": tokens}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="RNA Benchmark Data Preparation")
    parser.add_argument("--benchmark_dir", required=True, help="RNA benchmark data (.h5ad) directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--datasets",
        default="",
        help="Target dataset name (comma separated, all if empty)",
    )
    parser.add_argument("--max_cells_per_dataset", type=int, default=None, help="Maximum number of cells per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Sampling random number seed")
    parser.add_argument(
        "--gene_id_column",
        type=str,
        default=None,
        help="Gene ID column name in var of h5ad (automatically detected if not specified)",
    )
    parser.add_argument(
        "--gene_symbol_map",
        type=str,
        default=None,
        help="Gene symbol → Ensembl ID mapping TSV/CSV",
    )
    parser.add_argument(
        "--symbol_column",
        type=str,
        default="symbol",
        help="Symbol column name in mapping file",
    )
    parser.add_argument(
        "--ensembl_column",
        type=str,
        default="ensembl_id",
        help="Ensembl column name in mapping file",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_evaluation_logging(output_dir, "rna_benchmark_data_preparation")
    logger.info("=== RNA Benchmark Data Preparation Started ===")
    logger.info(f"Benchmark dir: {benchmark_dir}")
    logger.info(f"Output dir: {output_dir}")

    all_files = _discover_h5ad_files(benchmark_dir)
    selected_names = [name for name in args.datasets.split(",") if name.strip()]
    datasets = _select_datasets(all_files, selected_names if selected_names else None)

    if not datasets:
        raise FileNotFoundError("Target dataset not found.")

    tokenizer = TranscriptomeTokenizer()
    output_file = output_dir / "rna_benchmark_dataset.jsonl"
    if output_file.exists():
        logger.info(f"Append to existing file: {output_file}")

    summary: Dict[str, int] = {}
    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset.name} ({dataset.path})")
        tokens_list = _tokenize_h5ad(
            tokenizer=tokenizer,
            dataset=dataset,
            max_cells=args.max_cells_per_dataset,
            seed=args.seed,
            gene_id_column=args.gene_id_column,
            gene_symbol_map=Path(args.gene_symbol_map) if args.gene_symbol_map else None,
            symbol_column=args.symbol_column,
            ensembl_column=args.ensembl_column,
            logger=logger,
        )
        count = _write_jsonl(output_file, dataset.name, tokens_list)
        summary[dataset.name] = count
        logger.info(f"Saved {count} cells for dataset: {dataset.name}")

    summary_file = output_dir / "rna_benchmark_data_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=== RNA Benchmark Data Preparation Completed ===")
    logger.info(f"Output JSONL: {output_file}")
    logger.info(f"Summary JSON: {summary_file}")


if __name__ == "__main__":
    main()
