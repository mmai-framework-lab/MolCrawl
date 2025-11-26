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
        raw_data_dir: 生データ保存ディレクトリ (COMPOUNDS_DIR)
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

    download_datasets_from_repo(llamol_dir)

    # マーカーファイル作成
    data_dir = os.path.join(raw_data_dir, "data")
    marker_file = Path(data_dir) / "llamol_download.marker"
    marker_file.touch()


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
