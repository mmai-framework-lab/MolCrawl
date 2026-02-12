#!/usr/bin/env python3
"""
RNA Benchmark Data Preparation Script

scRNA-seq (.h5ad) から BERT/GPT-2 で評価可能な
トークン列データ（JSONL）を生成します。
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

from rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer
from utils.evaluation_output import setup_evaluation_logging


@dataclass
class DatasetConfig:
    """データセット設定"""

    name: str
    path: Path


def _discover_h5ad_files(benchmark_dir: Path) -> Dict[str, Path]:
    """ベンチマークディレクトリ内の .h5ad を列挙する"""
    h5ad_files: Dict[str, Path] = {}
    for file_path in benchmark_dir.glob("*.h5ad"):
        h5ad_files[file_path.stem] = file_path
    return h5ad_files


def _select_datasets(all_files: Dict[str, Path], selected: Optional[List[str]]) -> List[DatasetConfig]:
    """対象データセットを選別する"""
    if not selected:
        return [DatasetConfig(name=k, path=v) for k, v in sorted(all_files.items())]

    selected_set = {name.strip() for name in selected}
    configs: List[DatasetConfig] = []
    for name in selected_set:
        if name not in all_files:
            raise FileNotFoundError(f"指定されたデータセットが見つかりません: {name}")
        configs.append(DatasetConfig(name=name, path=all_files[name]))
    return configs


def _resolve_gene_id_column(adata: ad.AnnData, preferred: Optional[str]) -> Tuple[str, List[str]]:
    """遺伝子IDカラムを特定する"""
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["ensembl_id", "gene_id", "gene_ids", "feature_id", "gene"])
    candidates.extend(list(adata.var.columns))
    for col in candidates:
        if col in adata.var.columns:
            return col, list(adata.var[col].astype(str))
    # フォールバック: var_names
    return "__var_names__", list(adata.var_names.astype(str))


def _auto_detect_mapping_columns(
    df: pd.DataFrame,
    symbol_column: Optional[str],
    ensembl_column: Optional[str],
) -> Tuple[str, str]:
    """マッピングファイルの列名を自動検出する"""
    symbol_candidates = [col for col in [symbol_column, "hgnc_symbol", "symbol", "gene_symbol"] if col]
    ensembl_candidates = [col for col in [ensembl_column, "ensembl_gene_id", "ensembl_id", "gene_id"] if col]

    resolved_symbol = next((col for col in symbol_candidates if col in df.columns), None)
    resolved_ensembl = next((col for col in ensembl_candidates if col in df.columns), None)

    if resolved_symbol is None or resolved_ensembl is None:
        raise ValueError(
            "マッピングファイルの列名を自動検出できませんでした。"
            f" 利用可能カラム: {list(df.columns)}"
        )

    return resolved_symbol, resolved_ensembl


def _load_symbol_to_ensembl_map(
    mapping_path: Path,
    symbol_column: Optional[str],
    ensembl_column: Optional[str],
) -> Dict[str, str]:
    """遺伝子シンボル→Ensembl IDのマッピングを読み込む"""
    if not mapping_path.exists():
        raise FileNotFoundError(f"マッピングファイルが見つかりません: {mapping_path}")

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
    """遺伝子IDの一致率を検証する"""
    if not gene_ids:
        raise ValueError(f"{dataset_name}: 遺伝子IDが空です。")

    match_count = sum(1 for gene_id in gene_ids if gene_id in known_gene_ids)
    match_ratio = match_count / max(len(gene_ids), 1)
    logger.info(f"{dataset_name}: gene_id match ratio = {match_ratio:.2%}")

    if match_ratio < 0.1:
        raise ValueError(
            f"{dataset_name}: 遺伝子IDの一致率が低すぎます（{match_ratio:.2%}）。"
            " Ensembl ID列が存在しない可能性があります。"
        )


def _compute_n_counts(X_view) -> np.ndarray:
    """Xから細胞ごとの総カウントを算出する"""
    if sp.issparse(X_view):
        return np.asarray(X_view.sum(axis=1)).reshape(-1)
    return np.asarray(X_view.sum(axis=1)).reshape(-1)


def _tokenize_anndata_with_gene_ids(
    tokenizer: TranscriptomeTokenizer,
    adata: ad.AnnData,
    gene_ids: List[str],
) -> List[List[int]]:
    """anndataオブジェクトをトークン列に変換する（gene_id列指定対応）"""
    has_n_counts: bool = "n_counts" in adata.obs.columns

    # コーディング/miRNA遺伝子のみ選択
    coding_mirna_mask = np.array([tokenizer.genelist_dict.get(gene_id, False) for gene_id in gene_ids])
    coding_mirna_loc = np.where(coding_mirna_mask)[0]
    if len(coding_mirna_loc) == 0:
        raise ValueError("有効な遺伝子IDが見つかりませんでした。")

    norm_factor_vector = np.array([tokenizer.gene_median_dict[gene_ids[i]] for i in coding_mirna_loc])
    coding_mirna_tokens = np.array([tokenizer.gene_token_dict[gene_ids[i]] for i in coding_mirna_loc])

    # filter_pass があれば対象を限定
    if "filter_pass" in adata.obs.columns:
        filter_pass_loc = np.where(adata.obs["filter_pass"].values == 1)[0]
    else:
        filter_pass_loc = np.arange(adata.shape[0])

    tokenized_cells: List[List[int]] = []
    chunk_size: int = 512
    target_sum: float = 10_000

    for i in range(0, len(filter_pass_loc), chunk_size):
        idx = filter_pass_loc[i : i + chunk_size]
        # backed anndataでは行・列の同時fancy indexが制限されるため、
        # まず行のみで取得し、その後に列を抽出する
        X_rows = adata[idx].X
        if sp.issparse(X_rows):
            X_view = X_rows[:, coding_mirna_loc]
        else:
            X_view = np.asarray(X_rows)[:, coding_mirna_loc]
        if has_n_counts:
            n_counts = adata[idx].obs["n_counts"].values[:, None]
        else:
            # n_counts が無い場合は、対象遺伝子の総カウントを算出して代用
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
    """h5ad を読み込み、細胞ごとのトークン列を生成する"""
    random.seed(seed)
    adata = ad.read_h5ad(dataset.path, backed="r")

    gene_id_col, gene_ids = _resolve_gene_id_column(adata, gene_id_column)
    logger.info(f"{dataset.name}: gene_id column = {gene_id_col}")

    # シンボル→Ensemblのマッピングが指定されている場合は変換
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

    # numpy配列などを通常のlist[int]に変換
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
    """JSONLとして追記保存する"""
    count = 0
    with output_file.open("a", encoding="utf-8") as f:
        for tokens in tokens_list:
            record = {"dataset": dataset_name, "tokens": tokens}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="RNA Benchmark Data Preparation")
    parser.add_argument("--benchmark_dir", required=True, help="RNAベンチマークデータ(.h5ad)のディレクトリ")
    parser.add_argument("--output_dir", required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--datasets",
        default="",
        help="対象データセット名（カンマ区切り、空なら全て）",
    )
    parser.add_argument("--max_cells_per_dataset", type=int, default=None, help="データセットごとの最大細胞数")
    parser.add_argument("--seed", type=int, default=42, help="サンプリングの乱数シード")
    parser.add_argument(
        "--gene_id_column",
        type=str,
        default=None,
        help="h5adのvarにある遺伝子ID列名（未指定なら自動検出）",
    )
    parser.add_argument(
        "--gene_symbol_map",
        type=str,
        default=None,
        help="遺伝子シンボル→Ensembl IDのマッピングTSV/CSV",
    )
    parser.add_argument(
        "--symbol_column",
        type=str,
        default="symbol",
        help="マッピングファイル内のシンボル列名",
    )
    parser.add_argument(
        "--ensembl_column",
        type=str,
        default="ensembl_id",
        help="マッピングファイル内のEnsembl列名",
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
        raise FileNotFoundError("対象データセットが見つかりませんでした。")

    tokenizer = TranscriptomeTokenizer()
    output_file = output_dir / "rna_benchmark_dataset.jsonl"
    if output_file.exists():
        logger.info(f"既存ファイルに追記します: {output_file}")

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
