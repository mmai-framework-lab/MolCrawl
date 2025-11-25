#!/usr/bin/env python3
"""
OMIM Real Data Processor
========================

実際のOMIMデータファイルを処理してゲノム配列評価用データセットを作成するモジュール
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, List, Optional
from datetime import datetime

# プロジェクトルートを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class OMIMRealDataProcessor:
    """OMIM実データ処理クラス"""

    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # 設定ファイル読み込み
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_dir = self.config["data_directories"]["base_dir"]
        self.cache_dir = self.config["data_directories"]["cache_dir"]
        self.processed_dir = self.config["data_directories"]["processed_dir"]

        # ディレクトリ作成
        for dir_path in [self.data_dir, self.cache_dir, self.processed_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.logger.info("OMIM Real Data Processor initialized")
        self.logger.info(f"Data directory: {self.data_dir}")

    def download_omim_files(self, force_download: bool = False) -> Dict[str, str]:
        """OMIMファイルをダウンロード"""
        downloaded_files = {}

        for file_key, file_info in self.config["omim_data_sources"].items():
            local_path = os.path.join(self.data_dir, os.path.basename(file_info["local_path"]))

            if os.path.exists(local_path) and not force_download:
                self.logger.info(f"File already exists: {local_path}")
                downloaded_files[file_key] = local_path
                continue

            self.logger.info(f"Downloading {file_info['description']}")
            self.logger.info(f"URL: {file_info['url']}")

            try:
                response = requests.get(file_info["url"], timeout=30)
                response.raise_for_status()

                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(response.text)

                self.logger.info(f"Downloaded: {local_path}")
                downloaded_files[file_key] = local_path

            except Exception as e:
                self.logger.error(f"Failed to download {file_key}: {e}")
                raise

        return downloaded_files

    def parse_mim2gene(self, file_path: str) -> pd.DataFrame:
        """mim2gene.txtファイルを解析"""
        self.logger.info("Parsing mim2gene.txt")

        data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 5:
                    data.append(
                        {
                            "mim_number": parts[0],
                            "mim_entry_type": parts[1],
                            "entrez_gene_id": parts[2] if parts[2] != "" else None,
                            "approved_gene_symbol": parts[3] if parts[3] != "" else None,
                            "ensembl_gene_id": parts[4] if parts[4] != "" else None,
                        }
                    )

        df = pd.DataFrame(data)
        self.logger.info(f"Parsed {len(df)} mim2gene entries")
        return df

    def parse_mim_titles(self, file_path: str) -> pd.DataFrame:
        """mimTitles.txtファイルを解析"""
        self.logger.info("Parsing mimTitles.txt")

        data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                # Format: Prefix MIM_Number Title
                parts = line.split("\t")
                if len(parts) >= 3:
                    prefix = parts[0]
                    mim_number = parts[1]
                    title = parts[2]

                    # 遺伝パターンを推定
                    inheritance_pattern = self._extract_inheritance_pattern(title)

                    data.append(
                        {
                            "prefix": prefix,
                            "mim_number": mim_number,
                            "title": title,
                            "inheritance_pattern": inheritance_pattern,
                            "is_phenotype": prefix in ["#", "%", "^", "*"],
                        }
                    )

        df = pd.DataFrame(data)
        self.logger.info(f"Parsed {len(df)} mim titles entries")
        return df

    def parse_genemap2(self, file_path: str) -> pd.DataFrame:
        """genemap2.txtファイルを解析"""
        self.logger.info("Parsing genemap2.txt")

        column_names = [
            "chromosome",
            "genomic_position_start",
            "genomic_position_end",
            "cyto_location",
            "computed_cyto_location",
            "mim_number",
            "gene_symbols",
            "gene_name",
            "approved_symbol",
            "entrez_gene_id",
            "ensembl_gene_id",
            "comments",
            "phenotypes",
            "mouse_gene_symbol_id",
        ]

        data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= len(column_names):
                    row_data = {}
                    for i, col in enumerate(column_names):
                        row_data[col] = parts[i] if i < len(parts) and parts[i] != "" else None
                    data.append(row_data)

        df = pd.DataFrame(data)
        self.logger.info(f"Parsed {len(df)} genemap2 entries")
        return df

    def parse_morbidmap(self, file_path: str) -> pd.DataFrame:
        """morbidmap.txtファイルを解析"""
        self.logger.info("Parsing morbidmap.txt")

        data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 4:
                    disorder = parts[0]
                    gene_symbols = parts[1]
                    mim_number = parts[2]
                    cyto_location = parts[3]

                    # 病原性を推定
                    pathogenicity = self._estimate_pathogenicity(disorder)

                    data.append(
                        {
                            "disorder": disorder,
                            "gene_symbols": gene_symbols,
                            "mim_number": mim_number,
                            "cyto_location": cyto_location,
                            "pathogenicity": pathogenicity,
                            "is_disease_causing": 1 if pathogenicity in ["pathogenic", "likely_pathogenic"] else 0,
                        }
                    )

        df = pd.DataFrame(data)
        self.logger.info(f"Parsed {len(df)} morbidmap entries")
        return df

    def _extract_inheritance_pattern(self, title: str) -> str:
        """タイトルから遺伝パターンを抽出"""
        title_lower = title.lower()

        if "autosomal dominant" in title_lower or "ad" in title_lower:
            return "autosomal_dominant"
        elif "autosomal recessive" in title_lower or "ar" in title_lower:
            return "autosomal_recessive"
        elif "x-linked" in title_lower or "xlr" in title_lower or "xld" in title_lower:
            return "x_linked"
        elif "mitochondrial" in title_lower or "maternal" in title_lower:
            return "mitochondrial"
        elif "complex" in title_lower or "multifactorial" in title_lower:
            return "complex"
        else:
            return "unknown"

    def _estimate_pathogenicity(self, disorder: str) -> str:
        """疾患名から病原性を推定"""
        disorder_lower = disorder.lower()

        # 重篤な疾患キーワード
        severe_keywords = [
            "cancer",
            "carcinoma",
            "tumor",
            "syndrome",
            "disease",
            "deficiency",
            "dystrophy",
            "atrophy",
            "degeneration",
        ]

        # 軽度な疾患キーワード
        mild_keywords = ["susceptibility", "predisposition", "variant", "polymorphism"]

        if any(keyword in disorder_lower for keyword in severe_keywords):
            return "pathogenic"
        elif any(keyword in disorder_lower for keyword in mild_keywords):
            return "likely_pathogenic"
        else:
            return "uncertain_significance"

    def generate_sequences_for_genes(self, gene_symbols: List[str], sequence_length: int = 100) -> Dict[str, str]:
        """遺伝子シンボルに基づいてダミー配列を生成"""
        sequences = {}

        for gene in gene_symbols:
            # 遺伝子名をシードとして使用し、再現可能な配列を生成
            seed = hash(gene) % (2**32)
            np.random.seed(seed)

            nucleotides = ["A", "T", "G", "C"]
            sequence = "".join(np.random.choice(nucleotides, sequence_length))
            sequences[gene] = sequence

        return sequences

    def create_evaluation_dataset(self, downloaded_files: Dict[str, str]) -> pd.DataFrame:
        """評価用データセットを作成"""
        self.logger.info("Creating evaluation dataset from OMIM real data")

        # 各ファイルを解析
        mim2gene_df = self.parse_mim2gene(downloaded_files["mim2gene"])
        mim_titles_df = self.parse_mim_titles(downloaded_files["mim_titles"])
        genemap2_df = self.parse_genemap2(downloaded_files["genemap2"])
        morbidmap_df = self.parse_morbidmap(downloaded_files["morbidmap"])

        # データを統合
        self.logger.info("Merging OMIM datasets")

        # morbidmapをベースにして他のデータを結合
        merged_df = morbidmap_df.copy()

        # mim2geneと結合
        merged_df = merged_df.merge(
            mim2gene_df[["mim_number", "approved_gene_symbol", "entrez_gene_id"]],
            on="mim_number",
            how="left",
            suffixes=("", "_mim2gene"),
        )

        # mimTitlesと結合
        merged_df = merged_df.merge(
            mim_titles_df[["mim_number", "inheritance_pattern", "is_phenotype"]],
            on="mim_number",
            how="left",
        )

        # genemap2と結合
        merged_df = merged_df.merge(
            genemap2_df[["mim_number", "chromosome", "gene_name", "phenotypes"]],
            on="mim_number",
            how="left",
            suffixes=("", "_genemap2"),
        )

        # データクリーニング
        merged_df = merged_df.dropna(subset=["gene_symbols"])
        merged_df = merged_df[merged_df["gene_symbols"] != ""]

        # 設定に基づくフィルタリング
        max_sequences = self.config["processing_options"]["max_sequences"]
        if len(merged_df) > max_sequences:
            merged_df = merged_df.sample(n=max_sequences, random_state=42)

        # 配列生成
        self.logger.info("Generating sequences for genes")
        unique_genes = []
        for gene_symbols in merged_df["gene_symbols"].unique():
            if pd.notna(gene_symbols):
                genes = [g.strip() for g in gene_symbols.split(",")]
                unique_genes.extend(genes)

        unique_genes = list(set(unique_genes))
        sequence_length = self.config["processing_options"]["sequence_length"]
        sequences = self.generate_sequences_for_genes(unique_genes, sequence_length)

        # 配列をデータフレームに追加
        def get_sequence_for_row(row):
            gene_symbols = row["gene_symbols"]
            if pd.notna(gene_symbols):
                first_gene = gene_symbols.split(",")[0].strip()
                return sequences.get(first_gene, "")
            return ""

        merged_df["sequence"] = merged_df.apply(get_sequence_for_row, axis=1)

        # 空の配列を除外
        merged_df = merged_df[merged_df["sequence"] != ""]

        self.logger.info(f"Created evaluation dataset with {len(merged_df)} entries")
        self.logger.info(f"Disease-causing variants: {merged_df['is_disease_causing'].sum()}")
        self.logger.info(f"Benign variants: {(merged_df['is_disease_causing'] == 0).sum()}")

        return merged_df


def process_omim_real_data(
    config_path: str,
    output_dir: str,
    existing_omim_dir: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    OMIM実データを処理してデータセットを作成

    Args:
        config_path: 設定ファイルパス
        output_dir: 出力ディレクトリ
        existing_omim_dir: 既存のOMIMデータディレクトリ（指定時はダウンロードをスキップ）
        force_download: 強制ダウンロードフラグ

    Returns:
        出力ファイルパス
    """
    # ログ設定
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"omim_real_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting OMIM real data processing")

    if existing_omim_dir:
        logger.info(f"Using existing OMIM directory: {existing_omim_dir}")

    try:
        # プロセッサ初期化
        processor = OMIMRealDataProcessor(config_path, logger)

        # ファイルダウンロードまたは既存ファイル使用
        if existing_omim_dir and os.path.isdir(existing_omim_dir):
            logger.info("Using existing OMIM files, skipping download")
            # 既存ファイルのパスを設定
            downloaded_files = {
                "mim2gene": os.path.join(existing_omim_dir, "mim2gene.txt"),
                "mim_titles": os.path.join(existing_omim_dir, "mimTitles.txt"),
                "genemap2": os.path.join(existing_omim_dir, "genemap2.txt"),
                "morbidmap": os.path.join(existing_omim_dir, "morbidmap.txt"),
            }

            # ファイルの存在確認
            for _key, path in downloaded_files.items():
                if not os.path.exists(path):
                    logger.warning(f"File not found: {path}")
        else:
            # ファイルダウンロード
            downloaded_files = processor.download_omim_files(force_download)

        # データセット作成
        dataset = processor.create_evaluation_dataset(downloaded_files)

        # 結果保存
        # output_dirが既にdata配下を指している想定
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "omim_real_evaluation_dataset.csv")
        dataset.to_csv(output_file, index=False)

        logger.info("OMIM real data processing completed")
        logger.info(f"Output file: {output_file}")

        return output_file

    except Exception as e:
        logger.error(f"OMIM real data processing failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process OMIM Real Data")
    parser.add_argument("--config", type=str, required=True, help="Path to OMIM real data config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--existing_omim_dir",
        type=str,
        default=None,
        help="Existing OMIM data directory (skip download)",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force download even if files exist",
    )

    args = parser.parse_args()

    try:
        output_file = process_omim_real_data(
            config_path=args.config,
            output_dir=args.output_dir,
            existing_omim_dir=args.existing_omim_dir,
            force_download=args.force_download,
        )
        print(f"OMIM real data processing completed: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
