#!/usr/bin/env python3
"""
Protein Classification Data Preparation

タンパク質変異分類評価用のデータセット準備スクリプト
データ準備・評価・可視化の分離原則に基づいて作成
"""

import argparse
import logging
import os
import sys
from importlib import import_module

import numpy as np
import pandas as pd

# プロジェクトルートを設定して共通モジュールをインポート

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

check_learning_source_dir = import_module("utils.environment_check").check_learning_source_dir

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_default_output_dir():
    """デフォルトの出力ディレクトリを取得"""
    learning_source_dir = check_learning_source_dir()
    return os.path.join(learning_source_dir, "protein_sequence", "data", "protein_classification")


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    タンパク質変異データセットのサンプルを作成

    Args:
        output_path: 出力先CSVファイルパス
        num_samples: 生成するサンプル数

    Returns:
        生成されたデータセットのパス
    """
    logger.info(f"Creating sample protein classification dataset with {num_samples} samples")

    # サンプルタンパク質配列（様々な長さとタイプ）
    sample_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MGSSHHHHHHSSGLVPRGSHMKELKRLTCCKVQTCLRPPGQRQELAYFFKALPQCCNLCSPLVQNPKNCT",
        "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEL",
        "MADEAAQGAFQPGASGSRSKELKEAEDEAEEAEEAKEAEEEAKEAEEEAKEAEEEAKEAEEEA",
        "MGKEKIFSDDVRAIKEQKMLQIKHTAMAEVFLEQLACKMYSVDANTIKDFDLQHIWWNTVEQCE",
    ]

    data = []
    np.random.seed(42)

    for i in range(num_samples):
        # ランダムに配列を選択
        sequence = np.random.choice(sample_sequences)

        # ランダムに変異位置を選択（開始・終端を避ける）
        variant_pos = np.random.randint(5, len(sequence) - 5)
        ref_aa = sequence[variant_pos]

        # ランダムに代替アミノ酸を選択
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        alt_aa = np.random.choice([aa for aa in amino_acids if aa != ref_aa])

        # 病原性をルールベースで割り当て（デモ用）
        # 実際のデータはClinVarなどのデータベースから取得
        pathogenic = 0

        # 簡易ヒューリスティック: 特定のアミノ酸変異をより病原性が高いとする
        if ref_aa in "CGHPWY" and alt_aa not in "ACDEFGHIKLMNPQRSTVWY"[:10]:
            pathogenic = 1
        elif variant_pos < len(sequence) * 0.3:  # N末端領域
            pathogenic = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            pathogenic = np.random.choice([0, 1], p=[0.8, 0.2])

        data.append(
            {
                "variant_id": f"VAR_{i:03d}",
                "sequence": sequence,
                "variant_pos": variant_pos,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "pathogenic": pathogenic,
                "description": f"{ref_aa}{variant_pos + 1}{alt_aa}",
            }
        )

    # DataFrameに変換
    df = pd.DataFrame(data)

    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # CSVに保存
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Sample dataset created: {output_path}")
    logger.info(f"   Total samples: {len(df)}")
    logger.info(f"   Pathogenic: {df['pathogenic'].sum()}")
    logger.info(f"   Benign: {len(df) - df['pathogenic'].sum()}")

    return output_path


def prepare_protein_classification_data(input_csv=None, output_dir=None, num_samples=100, create_sample=False):
    """
    タンパク質分類データの準備メイン処理

    Args:
        input_csv: 入力CSVファイル（既存データを使用する場合）
        output_dir: 出力ディレクトリ
        num_samples: サンプルデータ生成時のサンプル数
        create_sample: サンプルデータを生成するかどうか

    Returns:
        準備されたデータセットのパス
    """
    if output_dir is None:
        output_dir = get_default_output_dir()

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    if create_sample:
        # サンプルデータ生成
        output_path = os.path.join(output_dir, "protein_classification_sample.csv")
        return create_sample_dataset(output_path, num_samples)

    elif input_csv:
        # 既存データを処理（現在は単純コピー、将来的には前処理を追加）
        logger.info(f"Processing existing data: {input_csv}")
        df = pd.read_csv(input_csv)

        # 基本的な検証
        required_columns = ["variant_id", "sequence", "pathogenic"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        output_path = os.path.join(output_dir, "protein_classification_processed.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Data processed: {output_path}")
        logger.info(f"   Total samples: {len(df)}")

        return output_path

    else:
        raise ValueError("Either --create_sample or --input_csv must be specified")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Protein Classification Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # サンプルデータ生成（100サンプル）
  python protein_classification_data_preparation.py --create_sample

  # カスタムサンプル数でデータ生成
  python protein_classification_data_preparation.py --create_sample --num_samples 500

  # 既存データの処理
  python protein_classification_data_preparation.py --input_csv data.csv --output_dir ./processed
""",
    )

    parser.add_argument("--input_csv", type=str, help="Input CSV file with protein variant data")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: $LEARNING_SOURCE_DIR/protein_sequence/data/protein_classification)",
    )

    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset for testing")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )

    args = parser.parse_args()

    try:
        dataset_path = prepare_protein_classification_data(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            create_sample=args.create_sample,
        )

        logger.info("=" * 70)
        logger.info("✅ Data preparation completed successfully")
        logger.info(f"📁 Dataset: {dataset_path}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
