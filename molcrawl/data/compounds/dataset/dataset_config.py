"""
Defining and configuring compound datasets

Provides a uniform definition for processing each dataset individually.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List


class CompoundDatasetType(str, Enum):
    """Compound dataset type"""

    ZINC20 = "zinc20"
    OPV = "opv"
    PC9_GAP = "pc9_gap"
    ZINC_QM9 = "zinc_qm9"
    REDDB = "reddb"
    CHEMBL = "chembl"
    PUBCHEMQC_2017 = "pubchemqc_2017"
    PUBCHEMQC_2020 = "pubchemqc_2020"
    GUACAMOL = "guacamol"  # Benchmark dataset for GPT2


@dataclass
class DatasetInfo:
    """Dataset information"""

    name: str  # dataset name
    dataset_type: CompoundDatasetType  # Dataset type
    source_subdir: str  # Raw data subdirectory (under data/)
    source_filename: str  # Raw data file name
    requires_download: bool = True  # Is download required?
    requires_properties: bool = True  # Is property calculation required?
    sample_size: Optional[int] = None  # Sampling size (if None, all data will be used)

    def get_raw_path(self, compounds_dir: Path) -> Path:
        """Get raw data path"""
        return compounds_dir / "data" / self.source_subdir / self.source_filename

    def get_processed_path(self, compounds_dir: Path) -> Path:
        """Get path to processed data"""
        return compounds_dir / "processed" / f"{self.name}.parquet"

    def get_tokenized_path(self, compounds_dir: Path) -> Path:
        """Get path to tokenized data"""
        return compounds_dir / "tokenized" / f"{self.name}_tokenized.parquet"

    def get_hf_dataset_path(self, compounds_dir: Path) -> Path:
        """Get the path in HuggingFace Dataset format"""
        return compounds_dir / "hf_datasets" / self.name


# Define each dataset
DATASET_DEFINITIONS = {
    CompoundDatasetType.ZINC20: DatasetInfo(
        name="zinc20",
        dataset_type=CompoundDatasetType.ZINC20,
        source_subdir="zinc20",
        source_filename="zinc_processed.parquet",
        requires_properties=True,
        sample_size=5_000_000,  # ZINC20 is large, so sample to 5M
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
        # logp and sascore are pre-computed in the source parquet; no recalculation needed.
        requires_properties=False,
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
        source_filename="guacamol_v1_train.smiles",  # Specify only train as representative
        requires_properties=False,  # No need to calculate physical properties because it is benchmark data
    ),
}


def get_dataset_info(dataset_type: CompoundDatasetType) -> DatasetInfo:
    """Get dataset information"""
    return DATASET_DEFINITIONS[dataset_type]


def get_available_datasets(compounds_dir: Path) -> List[CompoundDatasetType]:
    """
    Get list of available datasets

    Returns only datasets with raw data.

    Args:
        compounds_dir: compounds directorypath of

    Returns:
        List of available datasets
    """
    available = []
    for dataset_type, info in DATASET_DEFINITIONS.items():
        raw_path = info.get_raw_path(compounds_dir)
        if raw_path.exists():
            available.append(dataset_type)
    return available


def get_all_dataset_types() -> List[CompoundDatasetType]:
    """Get list of all dataset types"""
    return list(CompoundDatasetType)
