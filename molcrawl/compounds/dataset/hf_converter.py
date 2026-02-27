"""
HuggingFace Dataset形式への変換

トークナイズ済みデータをHuggingFace Dataset形式に変換します。
"""

import logging
from pathlib import Path
from typing import Optional, List

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

from molcrawl.compounds.dataset.dataset_config import DatasetInfo, CompoundDatasetType

logger = logging.getLogger(__name__)


class HFDatasetConverter:
    """
    HuggingFace Dataset変換クラス

    トークナイズ済みデータをHuggingFace Dataset形式に変換します。
    """

    def __init__(self, dataset_info: DatasetInfo, compounds_dir: Path):
        """
        Args:
            dataset_info: データセット情報
            compounds_dir: compoundsディレクトリのパス
        """
        self.dataset_info = dataset_info
        self.compounds_dir = Path(compounds_dir)

    def convert(
        self,
        train_ratio: float = 0.9,
        valid_ratio: float = 0.05,
        test_ratio: float = 0.05,
        force: bool = False,
        random_seed: int = 42,
    ) -> Optional[DatasetDict]:
        """
        HuggingFace Dataset形式に変換

        Args:
            train_ratio: 訓練データの割合
            valid_ratio: 検証データの割合
            test_ratio: テストデータの割合
            force: 強制再変換フラグ
            random_seed: ランダムシード

        Returns:
            DatasetDict（エラー時はNone）
        """
        hf_path = self.dataset_info.get_hf_dataset_path(self.compounds_dir)

        # 既に変換済みの場合はスキップ
        if not force and hf_path.exists():
            try:
                logger.info(f"✓ {self.dataset_info.name}: Already converted, loading from {hf_path}")
                # train/valid/testの全てが存在するか確認
                train_path = hf_path / "train"
                valid_path = hf_path / "valid"
                test_path = hf_path / "test"

                if train_path.exists() and valid_path.exists() and test_path.exists():
                    return DatasetDict(
                        {
                            "train": Dataset.load_from_disk(str(train_path)),
                            "valid": Dataset.load_from_disk(str(valid_path)),
                            "test": Dataset.load_from_disk(str(test_path)),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load existing dataset: {e}, will reconvert")

        # トークナイズ済みデータを読み込み
        tokenized_path = self.dataset_info.get_tokenized_path(self.compounds_dir)
        if not tokenized_path.exists():
            logger.warning(
                f"⚠ {self.dataset_info.name}: Tokenized data not found at {tokenized_path}\n  Please run tokenization first."
            )
            return None

        logger.info(f"🔄 {self.dataset_info.name}: Converting to HuggingFace Dataset format...")

        try:
            # データ読み込み
            table = pq.read_table(tokenized_path)
            df = table.to_pandas()

            # データ分割
            df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            total_samples = len(df)

            train_end = int(total_samples * train_ratio)
            valid_end = train_end + int(total_samples * valid_ratio)

            train_df = df[:train_end]
            valid_df = df[train_end:valid_end]
            test_df = df[valid_end:]

            logger.info(f"  Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

            # HuggingFace Dataset形式に変換
            dataset_dict = DatasetDict(
                {
                    "train": Dataset.from_pandas(train_df, preserve_index=False),
                    "valid": Dataset.from_pandas(valid_df, preserve_index=False),
                    "test": Dataset.from_pandas(test_df, preserve_index=False),
                }
            )

            # 保存
            hf_path.mkdir(parents=True, exist_ok=True)

            # 各splitを個別に保存
            for split_name, split_dataset in dataset_dict.items():
                split_path = hf_path / split_name
                split_dataset.save_to_disk(str(split_path))
                logger.info(f"  Saved {split_name} to {split_path}")

            logger.info(f"✓ {self.dataset_info.name}: Converted to HuggingFace Dataset format")
            return dataset_dict

        except Exception as e:
            logger.error(f"✗ {self.dataset_info.name}: Conversion failed - {e}")
            return None


def convert_all_tokenized_datasets(
    compounds_dir: Path,
    dataset_types: Optional[List[CompoundDatasetType]] = None,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
    test_ratio: float = 0.05,
    force: bool = False,
    random_seed: int = 42,
) -> dict:
    """
    トークナイズ済みの全データセットをHuggingFace形式に変換

    Args:
        compounds_dir: compoundsディレクトリのパス
        dataset_types: 変換するデータセット種別のリスト（Noneの場合はトークナイズ済みの全て）
        train_ratio: 訓練データの割合
        valid_ratio: 検証データの割合
        test_ratio: テストデータの割合
        force: 強制再変換フラグ
        random_seed: ランダムシード

    Returns:
        {dataset_type: dataset_dict} の辞書
    """
    from molcrawl.compounds.dataset.dataset_config import get_dataset_info, DATASET_DEFINITIONS

    # 処理対象のデータセットを決定
    if dataset_types is None:
        # トークナイズ済みデータが存在するデータセットを取得
        dataset_types = []
        for dt, info in DATASET_DEFINITIONS.items():
            tokenized_path = info.get_tokenized_path(compounds_dir)
            if tokenized_path.exists():
                dataset_types.append(dt)

        if not dataset_types:
            logger.warning("No tokenized datasets available for conversion")
            return {}
    else:
        # 指定されたデータセットが文字列の場合はEnumに変換
        if isinstance(dataset_types[0], str):
            dataset_types = [CompoundDatasetType(dt) for dt in dataset_types]

    logger.info(f"Converting {len(dataset_types)} datasets: {[dt.value for dt in dataset_types]}")

    results = {}
    for dataset_type in dataset_types:
        info = get_dataset_info(dataset_type)
        converter = HFDatasetConverter(info, compounds_dir)

        dataset_dict = converter.convert(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            force=force,
            random_seed=random_seed,
        )

        if dataset_dict is not None:
            results[dataset_type] = dataset_dict

    logger.info(f"Successfully converted {len(results)}/{len(dataset_types)} datasets")
    return results
