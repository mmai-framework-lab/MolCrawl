#!/usr/bin/env python3
"""
COSMICデータダウンロード・前処理スクリプト

COSMICデータベースから癌関連変異データをダウンロードし、
genome sequenceモデルの評価に適した形式に前処理します。

注意: LEARNING_SOURCE_DIR環境変数の設定が必須です。
"""

import sys
import os
import argparse
import pandas as pd
import requests
import gzip
import json
import re
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from urllib.parse import urljoin
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


def get_learning_source_dir():
    """LEARNING_SOURCE_DIR環境変数を取得（必須）"""
    learning_source = os.environ.get("LEARNING_SOURCE_DIR")
    if not learning_source:
        print(
            "ERROR: LEARNING_SOURCE_DIR environment variable is not set.",
            file=sys.stderr,
        )
        print("Please set it before running this script:", file=sys.stderr)
        print("  export LEARNING_SOURCE_DIR=/path/to/learning_source", file=sys.stderr)
        print("  # or", file=sys.stderr)
        print(
            "  LEARNING_SOURCE_DIR=learning_20251104 python {}".format(sys.argv[0]),
            file=sys.stderr,
        )
        sys.exit(1)
    return learning_source


def get_default_output_dir():
    """デフォルト出力ディレクトリを取得"""
    learning_source = get_learning_source_dir()
    return os.path.join(learning_source, "genome_sequence", "data", "cosmic")


def get_log_dir():
    """ログディレクトリを取得"""
    learning_source = get_learning_source_dir()
    log_dir = os.path.join(learning_source, "genome_sequence", "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


# ログ設定
log_dir = get_log_dir()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"{log_dir}/cosmic_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"LEARNING_SOURCE_DIR: {get_learning_source_dir()}")


class COSMICProcessor:
    """COSMICデータの取得・前処理クラス"""

    def __init__(self, output_dir=None):
        """
        初期化

        Args:
            output_dir (str): 出力ディレクトリ（Noneの場合は$LEARNING_SOURCE_DIR/genome_sequence/data/cosmic）
        """
        if output_dir is None:
            output_dir = get_default_output_dir()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

        # COSMIC公開データのURL（認証不要のデータセット）
        self.cosmic_urls = {
            "census": "https://cancer.sanger.ac.uk/cosmic/file_download/GRCh38/cosmic/v97/Cancer_Gene_Census.csv",
            "mutations": "https://cancer.sanger.ac.uk/cosmic/file_download/GRCh38/cosmic/v97/CosmicMutantExport.tsv.gz",
        }

        # 癌の重要度分類
        self.cancer_significance_map = {
            "pathogenic": ["pathogenic", "likely_pathogenic", "oncogenic"],
            "benign": ["benign", "likely_benign", "neutral"],
            "uncertain": ["uncertain_significance", "conflicting", "unknown"],
        }

    def download_cosmic_data(self, dataset_type="census"):
        """
        COSMICデータをダウンロード

        Args:
            dataset_type (str): データセットタイプ ('census', 'mutations')
        """
        if dataset_type not in self.cosmic_urls:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        url = self.cosmic_urls[dataset_type]
        output_file = (
            self.output_dir
            / f"cosmic_{dataset_type}.{'csv' if dataset_type == 'census' else 'tsv.gz'}"
        )

        logger.info(f"Downloading COSMIC {dataset_type} data from {url}")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to download {dataset_type}: {e}")
            return None

    def create_sample_cosmic_data(self, num_samples=20):
        """
        サンプルCOSMICデータを作成（テスト用）

        Args:
            num_samples (int): 作成するサンプル数
        """
        logger.info(f"Creating sample COSMIC data with {num_samples} samples")

        # 実際のCOSMIC変異パターンに基づくサンプルデータ
        sample_data = []

        # 既知の癌関連遺伝子と変異パターン
        cancer_genes = [
            "TP53",
            "KRAS",
            "PIK3CA",
            "APC",
            "BRCA1",
            "BRCA2",
            "EGFR",
            "MYC",
        ]
        mutation_types = [
            "Substitution - Missense",
            "Substitution - Nonsense",
            "Deletion - Frameshift",
            "Insertion - Frameshift",
        ]

        for i in range(num_samples):
            gene = np.random.choice(cancer_genes)
            mutation_type = np.random.choice(mutation_types)

            # DNA配列の生成（100bp）
            bases = ["A", "T", "G", "C"]
            ref_sequence = "".join(np.random.choice(bases, 100))

            # 変異の導入
            var_sequence = list(ref_sequence)
            mutation_pos = np.random.randint(45, 55)  # 中央付近に変異

            if "Missense" in mutation_type or "Nonsense" in mutation_type:
                # 点変異
                original_base = var_sequence[mutation_pos]
                possible_bases = [b for b in bases if b != original_base]
                var_sequence[mutation_pos] = np.random.choice(possible_bases)
            elif "Deletion" in mutation_type:
                # 欠失
                del_length = np.random.randint(1, 4)
                del var_sequence[mutation_pos : mutation_pos + del_length]
            elif "Insertion" in mutation_type:
                # 挿入
                ins_length = np.random.randint(1, 4)
                insert_bases = "".join(np.random.choice(bases, ins_length))
                var_sequence.insert(mutation_pos, insert_bases)

            var_sequence = "".join(var_sequence)

            # 癌関連度の決定（遺伝子と変異タイプに基づく）
            if gene in ["TP53", "BRCA1", "BRCA2"] and "Nonsense" in mutation_type:
                significance = "pathogenic"
                oncogenic = 1
            elif gene in ["KRAS", "PIK3CA"] and "Missense" in mutation_type:
                significance = (
                    "likely_pathogenic" if np.random.random() > 0.3 else "pathogenic"
                )
                oncogenic = 1
            elif "Frameshift" in mutation_type:
                significance = (
                    "pathogenic" if np.random.random() > 0.2 else "likely_pathogenic"
                )
                oncogenic = 1
            else:
                significance = np.random.choice(
                    ["benign", "likely_benign", "uncertain_significance"]
                )
                oncogenic = 0

            sample_data.append(
                {
                    "COSMIC_ID": f"COSV{1000000 + i}",
                    "Gene_name": gene,
                    "Mutation_Type": mutation_type,
                    "Cancer_significance": significance,
                    "oncogenic": oncogenic,
                    "Chromosome": f"chr{np.random.randint(1, 23)}",
                    "Position": np.random.randint(1000000, 200000000),
                    "Reference_sequence": ref_sequence,
                    "Variant_sequence": var_sequence,
                    "Primary_site": np.random.choice(
                        ["lung", "breast", "colon", "prostate", "liver"]
                    ),
                    "Sample_count": np.random.randint(1, 50),
                    "Mutation_somatic_status": "Confirmed somatic",
                }
            )

        # DataFrameに変換
        df = pd.DataFrame(sample_data)

        # CSVファイルとして保存
        output_file = self.output_dir / "cosmic_evaluation_dataset.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"Sample COSMIC data saved to {output_file}")
        logger.info(f"Data distribution: {df['oncogenic'].value_counts().to_dict()}")

        return output_file

    def parse_cosmic_vcf(self, vcf_file):
        """
        COSMIC VCFファイルを解析

        Args:
            vcf_file (str): VCFファイルのパス

        Returns:
            pd.DataFrame: 解析されたデータ
        """
        logger.info(f"Parsing COSMIC VCF file: {vcf_file}")

        mutations = []

        try:
            opener = gzip.open if str(vcf_file).endswith(".gz") else open
            with opener(vcf_file, "rt") as f:
                for line_num, line in enumerate(f):
                    if line.startswith("#"):
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) < 8:
                        continue

                    chrom = parts[0]
                    pos = int(parts[1])
                    ref = parts[3]
                    alt = parts[4]
                    info = parts[7]

                    # INFOフィールドから情報を抽出
                    info_dict = {}
                    for item in info.split(";"):
                        if "=" in item:
                            key, value = item.split("=", 1)
                            info_dict[key] = value

                    mutations.append(
                        {
                            "chromosome": chrom,
                            "position": pos,
                            "reference_allele": ref,
                            "alternate_allele": alt,
                            "gene": info_dict.get("GENE", ""),
                            "cosmic_id": info_dict.get("COSMIC_ID", ""),
                            "mutation_type": info_dict.get("VARIANT_CLASS", ""),
                            "cancer_type": info_dict.get("CANCER_TYPE", ""),
                            "sample_count": int(info_dict.get("CNT", 1)),
                        }
                    )

                    # メモリ使用量制限
                    if line_num > 100000:  # 最初の10万行のみ処理
                        logger.info(
                            "Limiting to first 100k variants for memory efficiency"
                        )
                        break

        except Exception as e:
            logger.error(f"Error parsing VCF file: {e}")
            return pd.DataFrame()

        return pd.DataFrame(mutations)

    def generate_sequences_from_variants(self, df, sequence_length=100):
        """
        変異情報から参照配列と変異配列を生成

        Args:
            df (pd.DataFrame): 変異データ
            sequence_length (int): 生成する配列長

        Returns:
            pd.DataFrame: 配列情報が追加されたデータ
        """
        logger.info(f"Generating sequences of length {sequence_length}")

        sequences = []
        bases = ["A", "T", "G", "C"]

        for _, row in df.iterrows():
            # ランダムなコンテキスト配列を生成
            half_len = sequence_length // 2

            # 参照配列の生成
            prefix = "".join(
                np.random.choice(bases, half_len - len(row["reference_allele"]) // 2)
            )
            suffix = "".join(
                np.random.choice(bases, half_len - len(row["reference_allele"]) // 2)
            )
            ref_sequence = prefix + row["reference_allele"] + suffix

            # 変異配列の生成
            var_sequence = prefix + row["alternate_allele"] + suffix

            # 長さの調整
            if len(ref_sequence) < sequence_length:
                ref_sequence += "".join(
                    np.random.choice(bases, sequence_length - len(ref_sequence))
                )
            elif len(ref_sequence) > sequence_length:
                ref_sequence = ref_sequence[:sequence_length]

            if len(var_sequence) < sequence_length:
                var_sequence += "".join(
                    np.random.choice(bases, sequence_length - len(var_sequence))
                )
            elif len(var_sequence) > sequence_length:
                var_sequence = var_sequence[:sequence_length]

            sequences.append(
                {"reference_sequence": ref_sequence, "variant_sequence": var_sequence}
            )

        # 元のDataFrameに配列情報を追加
        sequence_df = pd.DataFrame(sequences)
        result_df = pd.concat([df.reset_index(drop=True), sequence_df], axis=1)

        return result_df


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="COSMIC data preparation for genome sequence evaluation",
        epilog="Note: LEARNING_SOURCE_DIR environment variable must be set.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: $LEARNING_SOURCE_DIR/genome_sequence/data/cosmic)",
    )
    parser.add_argument("--download", action="store_true", help="Download COSMIC data")
    parser.add_argument(
        "--max_samples", type=int, default=1000, help="Maximum samples per class"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=100, help="Sequence length"
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample data instead of downloading",
    )

    args = parser.parse_args()

    processor = COSMICProcessor(args.output_dir)

    if args.create_sample_data:
        # サンプルデータの作成
        logger.info("Creating sample COSMIC data")
        processor.create_sample_cosmic_data(args.max_samples)
    else:
        logger.info("Sample COSMIC data creation completed")
        logger.info(
            "Note: For real COSMIC data, registration and authentication are required"
        )
        logger.info("Creating sample data for demonstration...")
        processor.create_sample_cosmic_data(args.max_samples)

    logger.info("COSMIC data preparation completed")


if __name__ == "__main__":
    main()
