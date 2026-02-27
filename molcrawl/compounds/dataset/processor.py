"""
個別データセット処理プロセッサー

各データセットを独立して処理するためのクラスを提供します。
"""

import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import pandas as pd
from molcrawl.compounds.dataset.dataset_config import CompoundDatasetType, DatasetInfo

logger = logging.getLogger(__name__)


def _get_rdkit_helpers():
    """RDKitヘルパー関数を取得"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Contrib.SA_Score import sascorer

    return Chem, Descriptors, sascorer


def calcLogPIfMol(smi):
    """LogP値を計算"""
    Chem, Descriptors, _ = _get_rdkit_helpers()
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return Descriptors.MolLogP(m)
    else:
        return None


def calcMolWeight(smi):
    """分子量を計算"""
    Chem, Descriptors, _ = _get_rdkit_helpers()
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Descriptors.ExactMolWt(mol)
    else:
        return None


def calcSascore(smi):
    """SA scoreを計算"""
    Chem, _, sascorer = _get_rdkit_helpers()
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return sascorer.calculateScore(mol)
    else:
        return None


class DatasetProcessor:
    """
    個別データセット処理クラス

    各データセットを独立して処理します：
    1. 生データの読み込み
    2. 物性計算（必要な場合）
    3. 処理済みデータの保存
    """

    def __init__(self, dataset_info: DatasetInfo, compounds_dir: Path, num_processes: int = 16):
        """
        Args:
            dataset_info: データセット情報
            compounds_dir: compoundsディレクトリのパス
            num_processes: 並列処理のプロセス数
        """
        self.dataset_info = dataset_info
        self.compounds_dir = Path(compounds_dir)
        self.num_processes = num_processes

    def process(self, force: bool = False) -> Optional[pd.DataFrame]:
        """
        データセットを処理

        Args:
            force: 強制再処理フラグ

        Returns:
            処理済みDataFrame（エラー時はNone）
        """
        processed_path = self.dataset_info.get_processed_path(self.compounds_dir)

        # 既に処理済みの場合はスキップ
        if not force and processed_path.exists():
            logger.info(f"✓ {self.dataset_info.name}: Already processed, skipping")
            return pd.read_parquet(processed_path)

        # 生データを読み込み
        df = self._load_raw_data()
        if df is None:
            return None

        # サンプリング
        if self.dataset_info.sample_size is not None and len(df) > self.dataset_info.sample_size:
            logger.info(f"  Sampling {self.dataset_info.sample_size} from {len(df)} samples")
            df = df.sample(n=self.dataset_info.sample_size, random_state=42)

        # 物性計算
        if self.dataset_info.requires_properties:
            df = self._calculate_properties(df)
            if df is None:
                return None

        # 保存
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path, index=False)
        logger.info(f"✓ {self.dataset_info.name}: Saved {len(df)} samples to {processed_path}")

        return df

    def _load_raw_data(self) -> Optional[pd.DataFrame]:
        """生データを読み込み"""
        raw_path = self.dataset_info.get_raw_path(self.compounds_dir)

        if not raw_path.exists():
            logger.warning(
                f"⚠ {self.dataset_info.name}: Raw data not found at {raw_path}\n  Please download this dataset first."
            )
            return None

        try:
            logger.info(f"📂 {self.dataset_info.name}: Loading from {raw_path}")
            df = pd.read_parquet(raw_path)

            # SMILES列が存在することを確認
            if "smiles" not in df.columns:
                logger.error(f"✗ {self.dataset_info.name}: 'smiles' column not found")
                return None

            logger.info(f"✓ {self.dataset_info.name}: Loaded {len(df)} samples")
            return df

        except Exception as e:
            logger.error(f"✗ {self.dataset_info.name}: Failed to load - {e}")
            return None

    def _calculate_properties(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """物性を計算"""
        logger.info(f"🧪 {self.dataset_info.name}: Calculating properties...")

        try:
            smi_series = df["smiles"]

            with multiprocessing.Pool(self.num_processes) as pool:
                # LogP計算
                logger.info("  Computing LogP...")
                logps_list = pool.map(calcLogPIfMol, smi_series)

                # 有効な分子のみをフィルタ
                valid_mols = ~pd.isna(logps_list)
                valid_smiles = smi_series[valid_mols].reset_index(drop=True)
                valid_logps = pd.Series(logps_list)[valid_mols].reset_index(drop=True)

                # 分子量計算
                logger.info("  Computing molecular weight...")
                mol_weights = pool.map(calcMolWeight, valid_smiles)

                # SA score計算
                logger.info("  Computing SA score...")
                sascores = pool.map(calcSascore, valid_smiles)

            # 結果をDataFrameに変換
            result_df = pd.DataFrame(
                {
                    "smiles": valid_smiles,
                    "logp": valid_logps,
                    "mol_weight": [w / 100.0 if w is not None else None for w in mol_weights],  # 正規化
                    "sascore": sascores,
                }
            )

            # NaNを含む行を削除
            initial_count = len(result_df)
            result_df = result_df.dropna()
            removed_count = initial_count - len(result_df)

            if removed_count > 0:
                logger.info(f"  Removed {removed_count} invalid molecules ({removed_count / initial_count * 100:.2f}%)")

            logger.info(f"✓ {self.dataset_info.name}: Properties calculated for {len(result_df)} molecules")
            return result_df

        except Exception as e:
            logger.error(f"✗ {self.dataset_info.name}: Property calculation failed - {e}")
            return None


def process_all_available_datasets(
    compounds_dir: Path,
    dataset_types: Optional[list] = None,
    force: bool = False,
    num_processes: int = 16,
) -> dict:
    """
    利用可能な全データセットを処理

    Args:
        compounds_dir: compoundsディレクトリのパス
        dataset_types: 処理するデータセット種別のリスト（Noneの場合は利用可能な全て）
        force: 強制再処理フラグ
        num_processes: 並列処理のプロセス数

    Returns:
        {dataset_type: processed_df} の辞書
    """
    from molcrawl.compounds.dataset.dataset_config import (
        get_available_datasets,
        get_dataset_info,
    )

    # 処理対象のデータセットを決定
    if dataset_types is None:
        # 利用可能な全データセットを取得
        available = get_available_datasets(compounds_dir)
        if not available:
            logger.warning("No datasets available for processing")
            return {}
        dataset_types = available
    else:
        # 指定されたデータセットが文字列の場合はEnumに変換
        if isinstance(dataset_types[0], str):
            dataset_types = [CompoundDatasetType(dt) for dt in dataset_types]

    logger.info(f"Processing {len(dataset_types)} datasets: {[dt.value for dt in dataset_types]}")

    results = {}
    for dataset_type in dataset_types:
        info = get_dataset_info(dataset_type)
        processor = DatasetProcessor(info, compounds_dir, num_processes)

        df = processor.process(force=force)
        if df is not None:
            results[dataset_type] = df

    logger.info(f"Successfully processed {len(results)}/{len(dataset_types)} datasets")
    return results
