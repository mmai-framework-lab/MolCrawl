import os
from pathlib import Path

from compounds.dataset.organix13.combine_all import combine_all
from compounds.dataset.organix13.download import download_datasets_from_repo
from compounds.dataset.organix13.opv.prepare_opv import OPV
from compounds.dataset.organix13.zinc.download_and_convert_to_parquet import (
    convert_zinc_to_parquet,
    download_zinc_files,
)


def download_zinc20(raw_data_dir: str):
    """
    ZINC20データセットのダウンロードと変換

    Args:
        raw_data_dir: 生データ保存ディレクトリ (COMPOUNDS_DIR)
    """
    download_zinc_files()
    # Save to data/zinc20 directory
    zinc_save_path = os.path.join(raw_data_dir, "data", "zinc20")
    convert_zinc_to_parquet(zinc_save_path)

    # マーカーファイル作成
    data_dir = os.path.join(raw_data_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    marker_file = Path(data_dir) / "zinc20_download.marker"
    marker_file.touch()


def download_opv(raw_data_dir: str):
    """
    OPVデータセットのダウンロード

    Args:
        raw_data_dir: 生データ保存ディレクトリ (COMPOUNDS_DIR)
    """
    # data/opvディレクトリに保存
    opv_dir = os.path.join(raw_data_dir, "data", "opv")
    os.makedirs(opv_dir, exist_ok=True)

    OPV(opv_dir)

    # マーカーファイル作成
    data_dir = os.path.join(raw_data_dir, "data")
    marker_file = Path(data_dir) / "opv_download.marker"
    marker_file.touch()


def download_llamol_datasets(raw_data_dir: str):
    """
    LlaMolデータセット（Fraunhofer-SCAI/llamolリポジトリからのダウンロード）

    Args:
        raw_data_dir: 生データ保存ディレクトリ (COMPOUNDS_DIR)
    """
    # data/Fraunhofer-SCAI-llamolディレクトリに保存
    llamol_dir = os.path.join(raw_data_dir, "data", "Fraunhofer-SCAI-llamol")
    os.makedirs(llamol_dir, exist_ok=True)

    # 既存のparquetファイルの整合性をチェック
    _verify_llamol_parquet_files(llamol_dir)

    download_datasets_from_repo(llamol_dir)

    # マーカーファイル作成
    data_dir = os.path.join(raw_data_dir, "data")
    marker_file = Path(data_dir) / "llamol_download.marker"
    marker_file.touch()


def _verify_llamol_parquet_files(llamol_dir: str):
    """
    LlaMolディレクトリ内のparquetファイルの整合性を検証

    Args:
        llamol_dir: LlaMolデータディレクトリ
    """
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)

    parquet_files = [
        "chembl_log_sascore.parquet",
        "qm9_zinc250_cep.parquet",
        "RedDB_Full.parquet",
        "pubchemqc_energy.parquet",
        "pubchemqc2020_energy.parquet",
    ]

    for filename in parquet_files:
        filepath = os.path.join(llamol_dir, filename)
        if os.path.exists(filepath):
            try:
                # ファイルの読み込みをテスト
                pd.read_parquet(filepath)
                logger.info(f"Verified: {filename} is valid")
            except Exception as e:
                logger.warning(
                    f"Corrupted parquet file detected: {filename}\nError: {e}\nDeleting corrupted file for re-download..."
                )
                try:
                    os.remove(filepath)
                    logger.info(f"Deleted corrupted file: {filepath}")
                except Exception as delete_error:
                    logger.error(f"Failed to delete corrupted file: {delete_error}")


def combine_datasets(raw_data_dir: str, output_dir: str):
    """
    全データセットを統合してOrganiX13を生成

    Args:
        raw_data_dir: 生データディレクトリ
        output_dir: 統合データ出力ディレクトリ
    """
    combine_all(raw_data_dir, output_dir)


# 後方互換性のためのエイリアス
def download_additional_datasets(raw_data_dir: str):
    """後方互換性のためのエイリアス。download_llamol_datasetsを使用してください。"""
    download_llamol_datasets(raw_data_dir)


def download_datasets(raw_data_dir: str, output_dir: str):
    """
    全データセットのダウンロードと統合（レガシー互換用）

    Args:
        raw_data_dir: 生データ保存ディレクトリ
        output_dir: 統合データ出力ディレクトリ
    """
    download_zinc20(raw_data_dir)
    download_opv(raw_data_dir)
    download_llamol_datasets(raw_data_dir)
    combine_datasets(raw_data_dir, output_dir)
