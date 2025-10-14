from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import numpy as np

import logging
import logging.config

from pathlib import Path

from core.base import read_parquet, save_parquet, multiprocess_tokenization, setup_logging
from compounds.utils.tokenizer import CompoundsTokenizer, ScaffoldsTokenizer
from compounds.utils.config import CompoundConfig
from compounds.utils.general import download_datasets

from config.paths import COMPOUNDS_DIR

logger = logging.getLogger(__name__)


def download_compound_datasets(cfg, organix13_dataset_path, download_marker, force=False):
    """
    化合物データセットのダウンロード処理

    Args:
        cfg: 設定オブジェクト
        organix13_dataset_path: データセット保存パス
        download_marker: ダウンロード完了マーカーファイル
        force: 強制再ダウンロードフラグ
    """
    if not force and download_marker.exists():
        logger.info("Dataset download already completed. Skipping download step.")
        return

    logger.info("Downloading datasets...")
    os.path.exists(cfg.raw_data_path) or os.makedirs(cfg.raw_data_path)
    download_datasets(cfg.raw_data_path, organix13_dataset_path)
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

    # SMILESのトークナイズ
    mol_tokenizer = CompoundsTokenizer(cfg.vocab_path, cfg.max_length)
    logger.info("Tokenizing SMILES...")
    processed_organix13 = multiprocess_tokenization(
        mol_tokenizer.bulk_tokenizer_parquet, organix13_dataset, column_name="smiles", new_column_name="tokens", processes=2
    )

    # Scaffoldsのトークナイズ
    scaffolds_tokenizer = ScaffoldsTokenizer(cfg.vocab_path, cfg.max_length)
    logger.info("Tokenizing Scaffolds...")
    processed_organix13 = multiprocess_tokenization(
        scaffolds_tokenizer.bulk_tokenizer_parquet,
        processed_organix13,
        column_name="smiles",
        new_column_name="scaffold_tokens",
    )

    logger.info("Tokenizing done.")
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
        plt.savefig("assets/img/compounds_tokenized_{}_lengths_dist.png".format(column_name))
        plt.close()
        logger.info(
            "Saved distribution of tokenized {} lengths to assets/img/compounds_tokenized_{}_lengths_dist.png".format(
                column_name, column_name
            )
        )

        return {
            "Number of Samples for {}".format(column_name): len(series_length),
            "Number of Tokens for {}".format(column_name): sum(series_length),
        }

    # 統計計算
    statistics = {**run_statistics(dataset["tokens"], "SMILES"), **run_statistics(dataset["scaffold_tokens"], "Scaffolds")}

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
    parser.add_argument("--force", action="store_true", help="Force re-download and reprocessing even if files exist")
    parser.add_argument("--download-only", action="store_true", help="Only perform download step")
    parser.add_argument("--tokenize-only", action="store_true", help="Only perform tokenization step")
    parser.add_argument("--stats-only", action="store_true", help="Only perform statistics step")
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
        download_compound_datasets(cfg, organix13_dataset_path, download_marker, args.force)

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
