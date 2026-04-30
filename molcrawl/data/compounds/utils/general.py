import os
from pathlib import Path

from molcrawl.data.compounds.dataset.organix13.combine_all import combine_all
from molcrawl.data.compounds.dataset.organix13.download import download_datasets_from_repo
from molcrawl.data.compounds.dataset.organix13.opv.prepare_opv import OPV
from molcrawl.data.compounds.dataset.organix13.zinc.download_and_convert_to_parquet import (
    convert_zinc_to_parquet,
    download_zinc_files,
)


def download_zinc20(raw_data_dir: str):
    """
    Download and convert ZINC20 dataset

    Args:
        raw_data_dir: raw data storage directory (COMPOUNDS_DIR)
    """
    download_zinc_files()
    # Save to data/zinc20 directory
    zinc_save_path = os.path.join(raw_data_dir, "data", "zinc20")
    convert_zinc_to_parquet(zinc_save_path)

    # Create marker file
    data_dir = os.path.join(raw_data_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    marker_file = Path(data_dir) / "zinc20_download.marker"
    marker_file.touch()


def download_opv(raw_data_dir: str):
    """
    Download OPV dataset

    Args:
        raw_data_dir: raw data storage directory (COMPOUNDS_DIR)
    """
    # save to data/opv directory
    opv_dir = os.path.join(raw_data_dir, "data", "opv")
    os.makedirs(opv_dir, exist_ok=True)

    OPV(opv_dir)

    # Create marker file
    data_dir = os.path.join(raw_data_dir, "data")
    marker_file = Path(data_dir) / "opv_download.marker"
    marker_file.touch()


def download_llamol_datasets(raw_data_dir: str):
    """
    LlaMol dataset (download from Fraunhofer-SCAI/llamol repository)

    Args:
        raw_data_dir: raw data storage directory (COMPOUNDS_DIR)
    """
    # Save in data/Fraunhofer-SCAI-llamol directory
    llamol_dir = os.path.join(raw_data_dir, "data", "Fraunhofer-SCAI-llamol")
    os.makedirs(llamol_dir, exist_ok=True)

    # Check the integrity of existing parquet files
    _verify_llamol_parquet_files(llamol_dir)

    download_datasets_from_repo(llamol_dir)

    # Create marker file
    data_dir = os.path.join(raw_data_dir, "data")
    marker_file = Path(data_dir) / "llamol_download.marker"
    marker_file.touch()


def _verify_llamol_parquet_files(llamol_dir: str):
    """
    Verify the integrity of parquet files in LlaMol directory

    Args:
        llamol_dir: LlaMol data directory
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
                # test file loading
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
    Combine all datasets to generate OrganiX13

    Args:
        raw_data_dir: raw data directory
        output_dir: Integrated data output directory
    """
    combine_all(raw_data_dir, output_dir)


# Alias ​​for backwards compatibility
def download_additional_datasets(raw_data_dir: str):
    """Alias ​​for backwards compatibility. Please use download_llamol_datasets."""
    download_llamol_datasets(raw_data_dir)


def download_datasets(raw_data_dir: str, output_dir: str):
    """
    Download and integrate all datasets (for legacy compatibility)

    Args:
        raw_data_dir: raw data storage directory
        output_dir: Integrated data output directory
    """
    download_zinc20(raw_data_dir)
    download_opv(raw_data_dir)
    download_llamol_datasets(raw_data_dir)
    combine_datasets(raw_data_dir, output_dir)
