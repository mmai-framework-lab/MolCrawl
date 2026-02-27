"""
化合物データセットの定義と設定

各データセットを個別に処理するための統一的な定義を提供します。
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List


class CompoundDatasetType(str, Enum):
    """化合物データセットの種類"""

    ZINC20 = "zinc20"
    OPV = "opv"
    PC9_GAP = "pc9_gap"
    ZINC_QM9 = "zinc_qm9"
    REDDB = "reddb"
    CHEMBL = "chembl"
    PUBCHEMQC_2017 = "pubchemqc_2017"
    PUBCHEMQC_2020 = "pubchemqc_2020"
    GUACAMOL = "guacamol"  # GPT2用のベンチマークデータセット


@dataclass
class DatasetInfo:
    """データセット情報"""

    name: str  # データセット名
    dataset_type: CompoundDatasetType  # データセット種別
    source_subdir: str  # 生データのサブディレクトリ (data/ 以下)
    source_filename: str  # 生データのファイル名
    requires_download: bool = True  # ダウンロードが必要か
    requires_properties: bool = True  # 物性計算が必要か
    sample_size: Optional[int] = None  # サンプリングサイズ（Noneの場合は全データ使用）

    def get_raw_path(self, compounds_dir: Path) -> Path:
        """生データのパスを取得"""
        return compounds_dir / "data" / self.source_subdir / self.source_filename

    def get_processed_path(self, compounds_dir: Path) -> Path:
        """処理済みデータのパスを取得"""
        return compounds_dir / "processed" / f"{self.name}.parquet"

    def get_tokenized_path(self, compounds_dir: Path) -> Path:
        """トークナイズ済みデータのパスを取得"""
        return compounds_dir / "tokenized" / f"{self.name}_tokenized.parquet"

    def get_hf_dataset_path(self, compounds_dir: Path) -> Path:
        """HuggingFace Dataset形式のパスを取得"""
        return compounds_dir / "hf_datasets" / self.name


# 各データセットの定義
DATASET_DEFINITIONS = {
    CompoundDatasetType.ZINC20: DatasetInfo(
        name="zinc20",
        dataset_type=CompoundDatasetType.ZINC20,
        source_subdir="zinc20",
        source_filename="zinc_processed.parquet",
        requires_properties=True,
        sample_size=5_000_000,  # ZINC20は大規模なので5Mにサンプリング
    ),
    CompoundDatasetType.OPV: DatasetInfo(
        name="opv",
        dataset_type=CompoundDatasetType.OPV,
        source_subdir="opv",
        source_filename="opv.parquet",
        requires_properties=True,
    ),
    CompoundDatasetType.PC9_GAP: DatasetInfo(
        name="pc9_gap",
        dataset_type=CompoundDatasetType.PC9_GAP,
        source_subdir="Fraunhofer-SCAI-llamol",
        source_filename="Full_PC9_GAP.parquet",
        requires_properties=True,
    ),
    CompoundDatasetType.ZINC_QM9: DatasetInfo(
        name="zinc_qm9",
        dataset_type=CompoundDatasetType.ZINC_QM9,
        source_subdir="Fraunhofer-SCAI-llamol",
        source_filename="qm9_zinc250_cep.parquet",
        requires_properties=True,
    ),
    CompoundDatasetType.REDDB: DatasetInfo(
        name="reddb",
        dataset_type=CompoundDatasetType.REDDB,
        source_subdir="Fraunhofer-SCAI-llamol",
        source_filename="RedDB_Full.parquet",
        requires_properties=True,
    ),
    CompoundDatasetType.CHEMBL: DatasetInfo(
        name="chembl",
        dataset_type=CompoundDatasetType.CHEMBL,
        source_subdir="Fraunhofer-SCAI-llamol",
        source_filename="chembl_log_sascore.parquet",
        requires_properties=True,
    ),
    CompoundDatasetType.PUBCHEMQC_2017: DatasetInfo(
        name="pubchemqc_2017",
        dataset_type=CompoundDatasetType.PUBCHEMQC_2017,
        source_subdir="Fraunhofer-SCAI-llamol",
        source_filename="pubchemqc_energy.parquet",
        requires_properties=True,
    ),
    CompoundDatasetType.PUBCHEMQC_2020: DatasetInfo(
        name="pubchemqc_2020",
        dataset_type=CompoundDatasetType.PUBCHEMQC_2020,
        source_subdir="Fraunhofer-SCAI-llamol",
        source_filename="pubchemqc2020_energy.parquet",
        requires_properties=True,
    ),
    CompoundDatasetType.GUACAMOL: DatasetInfo(
        name="guacamol",
        dataset_type=CompoundDatasetType.GUACAMOL,
        source_subdir="benchmark/GuacaMol",
        source_filename="guacamol_v1_train.smiles",  # trainのみを代表として指定
        requires_properties=False,  # ベンチマークデータなので物性計算不要
    ),
}


def get_dataset_info(dataset_type: CompoundDatasetType) -> DatasetInfo:
    """データセット情報を取得"""
    return DATASET_DEFINITIONS[dataset_type]


def get_available_datasets(compounds_dir: Path) -> List[CompoundDatasetType]:
    """
    利用可能なデータセットのリストを取得

    生データが存在するデータセットのみを返します。

    Args:
        compounds_dir: compoundsディレクトリのパス

    Returns:
        利用可能なデータセットのリスト
    """
    available = []
    for dataset_type, info in DATASET_DEFINITIONS.items():
        raw_path = info.get_raw_path(compounds_dir)
        if raw_path.exists():
            available.append(dataset_type)
    return available


def get_all_dataset_types() -> List[CompoundDatasetType]:
    """全データセット種別のリストを取得"""
    return list(CompoundDatasetType)
