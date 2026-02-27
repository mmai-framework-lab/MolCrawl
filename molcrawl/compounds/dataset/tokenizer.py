"""
個別データセットトークナイザー

各データセットを独立してトークナイズするためのクラスを提供します。
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from molcrawl.compounds.dataset.dataset_config import DatasetInfo, CompoundDatasetType
from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer, ScaffoldsTokenizer
from molcrawl.core.base import multiprocess_tokenization
from molcrawl.utils.image_manager import get_image_path

logger = logging.getLogger(__name__)


class DatasetTokenizer:
    """
    個別データセットトークナイザー

    処理済みデータセットをトークナイズします。
    """

    def __init__(
        self,
        dataset_info: DatasetInfo,
        compounds_dir: Path,
        vocab_path: str,
        max_length: int = 256,
        num_processes: int = 2,
    ):
        """
        Args:
            dataset_info: データセット情報
            compounds_dir: compoundsディレクトリのパス
            vocab_path: 語彙ファイルのパス
            max_length: 最大トークン長
            num_processes: 並列処理のプロセス数
        """
        self.dataset_info = dataset_info
        self.compounds_dir = Path(compounds_dir)
        self.vocab_path = vocab_path
        self.max_length = max_length
        self.num_processes = num_processes

        # トークナイザーの初期化
        self.mol_tokenizer = CompoundsTokenizer(vocab_path, max_length)
        self.scaffolds_tokenizer = ScaffoldsTokenizer(vocab_path, max_length)

    def tokenize(self, force: bool = False) -> Optional[pa.Table]:
        """
        データセットをトークナイズ

        Args:
            force: 強制再処理フラグ

        Returns:
            トークナイズ済みテーブル（エラー時はNone）
        """
        tokenized_path = self.dataset_info.get_tokenized_path(self.compounds_dir)

        # 既にトークナイズ済みの場合はスキップ
        if not force and tokenized_path.exists():
            logger.info(f"✓ {self.dataset_info.name}: Already tokenized, skipping")
            return pq.read_table(tokenized_path)

        # 処理済みデータを読み込み
        processed_path = self.dataset_info.get_processed_path(self.compounds_dir)
        if not processed_path.exists():
            logger.warning(
                f"⚠ {self.dataset_info.name}: Processed data not found at {processed_path}\n  Please run processing first."
            )
            return None

        logger.info(f"🔤 {self.dataset_info.name}: Tokenizing...")

        try:
            # データ読み込み
            table = pq.read_table(processed_path)

            # SMILESのトークナイズ
            logger.info("  Tokenizing SMILES...")
            table = multiprocess_tokenization(
                self.mol_tokenizer.bulk_tokenizer_parquet,
                table,
                column_name="smiles",
                new_column_name="tokens",
                processes=self.num_processes,
            )

            # Scaffoldsのトークナイズ
            logger.info("  Tokenizing Scaffolds...")
            table = multiprocess_tokenization(
                self.scaffolds_tokenizer.bulk_tokenizer_parquet,
                table,
                column_name="smiles",
                new_column_name="scaffold_tokens",
                processes=self.num_processes,
            )

            # 無効なSMILESの統計を出力
            self._report_invalid_smiles_stats()

            # 保存
            tokenized_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, tokenized_path)

            logger.info(f"✓ {self.dataset_info.name}: Tokenized {table.num_rows} samples")
            return table

        except Exception as e:
            logger.error(f"✗ {self.dataset_info.name}: Tokenization failed - {e}")
            return None

    def _report_invalid_smiles_stats(self):
        """無効なSMILESの統計をログに出力"""
        from molcrawl.compounds.utils.preprocessing import get_invalid_smiles_stats

        invalid_count, total_count, invalid_rate, examples = get_invalid_smiles_stats()
        if total_count == 0:
            return

        name = self.dataset_info.name
        logger.info(f"  [{name}] SMILES validation: {invalid_count}/{total_count} invalid ({invalid_rate:.2f}%)")

        if examples:
            logger.info(f"  [{name}] Examples of invalid SMILES:")
            for i, (reason, smiles) in enumerate(examples, 1):
                logger.info(f"    {i}. [{reason}] {smiles}")

        if invalid_rate > 10.0:
            logger.error(
                f"  [{name}] Very high invalid SMILES rate ({invalid_rate:.2f}%). "
                "Data quality issues should be investigated."
            )
        elif invalid_rate > 5.0:
            logger.warning(
                f"  [{name}] High invalid SMILES rate ({invalid_rate:.2f}%). " "This may indicate data quality issues."
            )
        elif invalid_rate > 1.0:
            logger.info(
                f"  [{name}] Moderate invalid SMILES rate ({invalid_rate:.2f}%). "
                "Within acceptable range for large chemical databases."
            )
        else:
            logger.info(f"  [{name}] Low invalid SMILES rate ({invalid_rate:.2f}%). Data quality is good.")


def compute_tokenization_statistics(
    compounds_dir: Path,
    dataset_types: Optional[List[CompoundDatasetType]] = None,
    force: bool = False,
) -> Dict[CompoundDatasetType, dict]:
    """
    トークナイズ済みデータセットの統計計算と可視化

    各データセットについてトークン長分布のヒストグラムを生成し、
    サンプル数・トークン数の統計を出力します。

    Args:
        compounds_dir: compoundsディレクトリのパス
        dataset_types: 統計を計算するデータセット種別のリスト（Noneの場合はトークナイズ済みの全て）
        force: 強制再計算フラグ

    Returns:
        {dataset_type: statistics_dict} の辞書
    """
    from molcrawl.compounds.dataset.dataset_config import get_dataset_info, DATASET_DEFINITIONS

    # 対象データセットを決定
    if dataset_types is None:
        dataset_types = []
        for dt, info in DATASET_DEFINITIONS.items():
            tokenized_path = info.get_tokenized_path(compounds_dir)
            if tokenized_path.exists():
                dataset_types.append(dt)

        if not dataset_types:
            logger.warning("No tokenized datasets available for statistics")
            return {}
    else:
        if isinstance(dataset_types[0], str):
            dataset_types = [CompoundDatasetType(dt) for dt in dataset_types]

    logger.info(f"Computing statistics for {len(dataset_types)} datasets: {[dt.value for dt in dataset_types]}")

    all_results = {}
    for dataset_type in dataset_types:
        info = get_dataset_info(dataset_type)
        tokenized_path = info.get_tokenized_path(compounds_dir)

        if not tokenized_path.exists():
            logger.warning(f"⚠ {info.name}: Tokenized data not found, skipping statistics")
            continue

        # マーカーファイルで冪等性を確保
        stats_marker = tokenized_path.parent / f"{info.name}_stats.marker"
        if not force and stats_marker.exists():
            logger.info(f"✓ {info.name}: Statistics already computed, skipping")
            continue

        logger.info(f"📊 {info.name}: Computing statistics...")

        try:
            table = pq.read_table(tokenized_path)
            statistics = {}

            for column_name, display_name in [("tokens", "SMILES"), ("scaffold_tokens", "Scaffolds")]:
                if column_name not in table.column_names:
                    logger.warning(f"  {info.name}: Column '{column_name}' not found, skipping")
                    continue

                series_length = []
                for item in table[column_name]:
                    if item.is_valid:
                        series_length.append(len(item))

                # ヒストグラム生成
                plt.figure()
                plt.hist(series_length, bins=np.arange(0, 200, 1))
                plt.xlabel(f"Length of tokenized {display_name}")
                plt.title(f"[{info.name}] Distribution of tokenized {display_name} lengths")

                image_path = get_image_path(
                    "compounds",
                    f"compounds_{info.name}_tokenized_{display_name}_lengths_dist.png",
                )
                plt.savefig(image_path)
                plt.close()
                logger.info(f"  Saved histogram to {image_path}")

                statistics[f"Number of Samples for {display_name}"] = len(series_length)
                statistics[f"Number of Tokens for {display_name}"] = sum(series_length)

            for key, value in statistics.items():
                logger.info(f"  {info.name}: {key}: {value}")

            stats_marker.touch()
            all_results[dataset_type] = statistics

        except Exception as e:
            logger.error(f"✗ {info.name}: Statistics computation failed - {e}")

    logger.info(f"Successfully computed statistics for {len(all_results)}/{len(dataset_types)} datasets")
    return all_results


def tokenize_all_processed_datasets(
    compounds_dir: Path,
    vocab_path: str,
    max_length: int = 256,
    dataset_types: Optional[List[CompoundDatasetType]] = None,
    force: bool = False,
    num_processes: int = 2,
) -> dict:
    """
    処理済みの全データセットをトークナイズ

    Args:
        compounds_dir: compoundsディレクトリのパス
        vocab_path: 語彙ファイルのパス
        max_length: 最大トークン長
        dataset_types: トークナイズするデータセット種別のリスト（Noneの場合は処理済みの全て）
        force: 強制再処理フラグ
        num_processes: 並列処理のプロセス数

    Returns:
        {dataset_type: tokenized_table} の辞書
    """
    from molcrawl.compounds.dataset.dataset_config import get_dataset_info, DATASET_DEFINITIONS

    # 処理対象のデータセットを決定
    if dataset_types is None:
        # 処理済みデータが存在するデータセットを取得
        dataset_types = []
        for dt, info in DATASET_DEFINITIONS.items():
            processed_path = info.get_processed_path(compounds_dir)
            if processed_path.exists():
                dataset_types.append(dt)

        if not dataset_types:
            logger.warning("No processed datasets available for tokenization")
            return {}
    else:
        # 指定されたデータセットが文字列の場合はEnumに変換
        if isinstance(dataset_types[0], str):
            dataset_types = [CompoundDatasetType(dt) for dt in dataset_types]

    logger.info(f"Tokenizing {len(dataset_types)} datasets: {[dt.value for dt in dataset_types]}")

    results = {}
    for dataset_type in dataset_types:
        info = get_dataset_info(dataset_type)
        tokenizer = DatasetTokenizer(info, compounds_dir, vocab_path, max_length, num_processes)

        table = tokenizer.tokenize(force=force)
        if table is not None:
            results[dataset_type] = table

    logger.info(f"Successfully tokenized {len(results)}/{len(dataset_types)} datasets")
    return results
