import logging
import logging.config
import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルートのsrcディレクトリをパスに追加

from compounds.utils.config import CompoundConfig
from utils.image_manager import get_image_path
from compounds.utils.general import (
    combine_datasets,
    download_datasets,
    download_llamol_datasets,
    download_opv,
    download_zinc20,
)
from compounds.utils.tokenizer import CompoundsTokenizer, ScaffoldsTokenizer
from config.paths import COMPOUNDS_DIR
from core.base import (
    multiprocess_tokenization,
    read_parquet,
    save_parquet,
    setup_logging,
)

logger = logging.getLogger(__name__)


def download_compound_datasets(cfg, organix13_dataset_path, download_marker, force=False, dataset_type="all"):
    """
    化合物データセットのダウンロード処理

    Args:
        cfg: 設定オブジェクト
        organix13_dataset_path: データセット保存パス
        download_marker: ダウンロード完了マーカーファイル
        force: 強制再ダウンロードフラグ
        dataset_type: ダウンロードするデータセット種別
                      ("all", "zinc20", "opv", "additional", "combine")
    """
    if not force and download_marker.exists():
        logger.info("Dataset download already completed. Skipping download step.")
        return

    logger.info(f"Downloading datasets (type: {dataset_type})...")
    os.path.exists(cfg.raw_data_path) or os.makedirs(cfg.raw_data_path)

    if dataset_type == "all":
        download_datasets(cfg.raw_data_path, organix13_dataset_path)
    elif dataset_type == "zinc20":
        logger.info("Downloading ZINC20 dataset...")
        download_zinc20(cfg.raw_data_path)
    elif dataset_type == "opv":
        logger.info("Downloading OPV dataset...")
        download_opv(cfg.raw_data_path)
    elif dataset_type == "llamol":
        logger.info("Downloading LlaMol datasets from Fraunhofer-SCAI/llamol repository...")
        download_llamol_datasets(cfg.raw_data_path)
    elif dataset_type == "combine":
        logger.info("Combining all datasets into OrganiX13...")
        combine_datasets(cfg.raw_data_path, organix13_dataset_path)
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be one of: all, zinc20, opv, llamol, combine")

    download_marker.touch()
    logger.info("Download completed.")


def tokenize_compound_data(cfg, organix13_dataset_path, tokenized_marker, processed_parquet, force=False):
    """
    化合物データのトークナイズ処理

    Args:
        cfg: 設定オブジェクト
        organix13_dataset_path: データセット保存パス
        tokenized_marker: トークナイズ完了マーカーファイル
        processed_parquet: 処理済みParquetファイルパス
        force: 強制再処理フラグ

    Returns:
        tokenized_dataset: トークナイズ済みデータセット
    """
    if not force and tokenized_marker.exists() and processed_parquet.exists():
        logger.info("Tokenization already completed. Skipping tokenization step.")
        return read_parquet(file_path=str(processed_parquet))

    # 元データの読み込み
    organix13_dataset = read_parquet(file_path=os.path.join(organix13_dataset_path, "OrganiX13.parquet"))

    # プロセス数を環境変数から取得（デフォルト: 2）
    num_processes = int(os.environ.get('TOKENIZATION_PROCESSES', '2'))
    logger.info(f"Using {num_processes} processes for tokenization")

    # SMILESのトークナイズ
    mol_tokenizer = CompoundsTokenizer(cfg.vocab_path, cfg.max_length)
    logger.info("Tokenizing SMILES...")
    processed_organix13 = multiprocess_tokenization(
        mol_tokenizer.bulk_tokenizer_parquet,
        organix13_dataset,
        column_name="smiles",
        new_column_name="tokens",
        processes=num_processes,
    )

    # Scaffoldsのトークナイズ
    scaffolds_tokenizer = ScaffoldsTokenizer(cfg.vocab_path, cfg.max_length)
    logger.info("Tokenizing Scaffolds...")
    processed_organix13 = multiprocess_tokenization(
        scaffolds_tokenizer.bulk_tokenizer_parquet,
        processed_organix13,
        column_name="smiles",
        new_column_name="scaffold_tokens",
        processes=num_processes,
    )

    logger.info("Tokenizing done.")

    # 無効なSMILESの統計を出力
    from compounds.utils.preprocessing import get_invalid_smiles_stats
    invalid_count, total_count, invalid_rate, examples = get_invalid_smiles_stats()
    if total_count > 0:
        logger.info(
            f"SMILES validation summary: {invalid_count}/{total_count} invalid SMILES ({invalid_rate:.2f}%)"
        )

        # 無効なSMILESの例を表示
        if examples:
            logger.info("Examples of invalid SMILES:")
            for i, (reason, smiles) in enumerate(examples, 1):
                logger.info(f"  {i}. [{reason}] {smiles}")

        # 無効率に基づく評価
        if invalid_rate > 10.0:
            logger.error(
                f"Very high rate of invalid SMILES detected ({invalid_rate:.2f}%). "
                "This indicates serious data quality issues that should be investigated."
            )
        elif invalid_rate > 5.0:
            logger.warning(
                f"High rate of invalid SMILES detected ({invalid_rate:.2f}%). "
                "This may indicate data quality issues."
            )
        elif invalid_rate > 1.0:
            logger.info(
                f"Moderate rate of invalid SMILES ({invalid_rate:.2f}%). "
                "This is within acceptable range for large chemical databases."
            )
        else:
            logger.info(
                f"Low rate of invalid SMILES ({invalid_rate:.2f}%). "
                "Data quality is good."
            )

        # ZINC20特有の問題の説明
        logger.info(
            "Note: ZINC20 may contain some invalid SMILES due to:\n"
            "  - Quaternary ammonium ions (N+) and other charged species\n"
            "  - Format conversion errors from other chemical representations\n"
            "  - Complex stereochemistry or unusual bonding patterns\n"
            "  - These are typically <5% of the dataset and are expected in large databases"
        )

    save_parquet(table=processed_organix13, file_path=processed_parquet)
    tokenized_marker.touch()

    return processed_organix13


def compute_tokenization_statistics(dataset, stats_marker, force=False):
    """
    トークナイズ統計の計算と可視化

    Args:
        dataset: トークナイズ済みデータセット
        stats_marker: 統計処理完了マーカーファイル
        force: 強制再計算フラグ
    """
    if not force and stats_marker.exists():
        logger.info("Statistics already computed. Skipping statistics step.")
        return

    logger.info("Computing Statistics...")

    def run_statistics(table_row, column_name):
        """個別カラムの統計処理"""
        series_length = []
        for i in table_row:
            if i.is_valid:
                series_length.append(len(i))

        plt.hist(series_length, bins=np.arange(0, 200, 1))
        plt.xlabel("Length of tokenized {}".format(column_name))
        plt.title("Distribution of tokenized {} lengths".format(column_name))

        image_path = get_image_path("compounds", "compounds_tokenized_{}_lengths_dist.png".format(column_name))
        plt.savefig(image_path)
        plt.close()
        logger.info(f"Saved distribution of tokenized {column_name} lengths to {image_path}")

        return {
            "Number of Samples for {}".format(column_name): len(series_length),
            "Number of Tokens for {}".format(column_name): sum(series_length),
        }

    # 統計計算
    statistics = {
        **run_statistics(dataset["tokens"], "SMILES"),
        **run_statistics(dataset["scaffold_tokens"], "Scaffolds"),
    }

    # 結果出力
    for key, value in statistics.items():
        logger.info("{}: {}".format(key, value))

    stats_marker.touch()


def main():
    """
    化合物データ準備のメイン実行関数
    """
    parser = ArgumentParser()
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reprocessing even if files exist",
    )
    parser.add_argument("--download-only", action="store_true", help="Only perform download step")
    parser.add_argument("--tokenize-only", action="store_true", help="Only perform tokenization step")
    parser.add_argument("--stats-only", action="store_true", help="Only perform statistics step")
    parser.add_argument(
        "--dataset-type",
        choices=["all", "zinc20", "opv", "llamol", "combine"],
        default="all",
        help="Dataset type to download: all (default), zinc20, opv, llamol, or combine",
    )
    args = parser.parse_args()

    # 設定とパスの初期化
    cfg = CompoundConfig.from_file(args.config).data_preparation
    organix13_dataset_path = COMPOUNDS_DIR + "/organix13"
    os.path.exists(organix13_dataset_path) or os.makedirs(organix13_dataset_path)

    setup_logging(COMPOUNDS_DIR + "/compounds_logs")

    # マーカーファイル・出力ファイル
    download_marker = Path(organix13_dataset_path) / "download_complete.marker"
    tokenized_marker = Path(organix13_dataset_path) / "tokenized_complete.marker"
    stats_marker = Path(organix13_dataset_path) / "stats_complete.marker"
    processed_parquet = Path(organix13_dataset_path) / "OrganiX13_tokenized.parquet"

    # 実行ステップの制御
    run_download = not args.tokenize_only and not args.stats_only
    run_tokenize = not args.download_only and not args.stats_only
    run_stats = not args.download_only and not args.tokenize_only

    # 1. データダウンロード
    if run_download:
        download_compound_datasets(
            cfg,
            organix13_dataset_path,
            download_marker,
            args.force,
            dataset_type=args.dataset_type,
        )

    # 2. トークナイズ処理
    organix13_dataset = None
    if run_tokenize:
        organix13_dataset = tokenize_compound_data(cfg, organix13_dataset_path, tokenized_marker, processed_parquet, args.force)

    # 3. 統計処理
    if run_stats:
        if organix13_dataset is None:
            # 統計のみ実行する場合はデータを読み込み
            if processed_parquet.exists():
                organix13_dataset = read_parquet(file_path=str(processed_parquet))
            else:
                logger.error("Tokenized data not found. Please run tokenization first.")
                return

        compute_tokenization_statistics(organix13_dataset, stats_marker, args.force)

    logger.info("Processing completed. Processed dataset saved to {}.".format(COMPOUNDS_DIR))


if __name__ == "__main__":
    main()
