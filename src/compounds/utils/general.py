import os
from pathlib import Path
from compounds.dataset.organix13.zinc.download_and_convert_to_parquet import (
    download_zinc_files,
    convert_zinc_to_parquet,
)
from compounds.dataset.organix13.combine_all import combine_all
from compounds.dataset.organix13.opv.prepare_opv import OPV
from compounds.dataset.organix13.download import download_datasets_from_repo


def download_zinc20(raw_data_dir: str):
    """
    ZINC20データセットのダウンロードと変換
    
    Args:
        raw_data_dir: 生データ保存ディレクトリ
    """
    download_zinc_files()
    convert_zinc_to_parquet(raw_data_dir)
    
    # マーカーファイル作成
    data_dir = os.path.join(raw_data_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    marker_file = Path(data_dir) / "zinc20_download.marker"
    marker_file.touch()


def download_opv(raw_data_dir: str):
    """
    OPVデータセットのダウンロード
    
    Args:
        raw_data_dir: 生データ保存ディレクトリ
    """
    OPV(raw_data_dir)
    
    # マーカーファイル作成
    data_dir = os.path.join(raw_data_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    marker_file = Path(data_dir) / "opv_download.marker"
    marker_file.touch()


def download_additional_datasets(raw_data_dir: str):
    """
    追加データセット（リポジトリからのダウンロード）
    
    Args:
        raw_data_dir: 生データ保存ディレクトリ
    """
    download_datasets_from_repo(raw_data_dir)
    
    # マーカーファイル作成
    data_dir = os.path.join(raw_data_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    marker_file = Path(data_dir) / "additional_download.marker"
    marker_file.touch()


def combine_datasets(raw_data_dir: str, output_dir: str):
    """
    全データセットを統合してOrganiX13を生成
    
    Args:
        raw_data_dir: 生データディレクトリ
        output_dir: 統合データ出力ディレクトリ
    """
    combine_all(raw_data_dir, output_dir)


def download_datasets(raw_data_dir: str, output_dir: str):
    """
    全データセットのダウンロードと統合（レガシー互換用）
    
    Args:
        raw_data_dir: 生データ保存ディレクトリ
        output_dir: 統合データ出力ディレクトリ
    """
    download_zinc20(raw_data_dir)
    download_opv(raw_data_dir)
    download_additional_datasets(raw_data_dir)
    combine_datasets(raw_data_dir, output_dir)
