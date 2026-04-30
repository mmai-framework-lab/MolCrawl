"""
Individual dataset processing processor

Provides classes for processing each dataset independently.
"""

import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import pandas as pd
from molcrawl.data.compounds.dataset.dataset_config import CompoundDatasetType, DatasetInfo

logger = logging.getLogger(__name__)


def _get_rdkit_helpers():
    """Get RDKit helper functions"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Contrib.SA_Score import sascorer

    return Chem, Descriptors, sascorer


def calcLogPIfMol(smi):
    """Calculate LogP value"""
    Chem, Descriptors, _ = _get_rdkit_helpers()
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return Descriptors.MolLogP(m)
    else:
        return None


def calcMolWeight(smi):
    """Calculate molecular weight"""
    Chem, Descriptors, _ = _get_rdkit_helpers()
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Descriptors.ExactMolWt(mol)
    else:
        return None


def calcSascore(smi):
    """Calculate SA score"""
    Chem, _, sascorer = _get_rdkit_helpers()
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return sascorer.calculateScore(mol)
    else:
        return None


class DatasetProcessor:
    """
    Individual dataset processing class

    Process each dataset independently:
    1. Loading raw data
    2. Physical property calculations (if necessary)
    3. Saving processed data
    """

    def __init__(self, dataset_info: DatasetInfo, compounds_dir: Path, num_processes: int = 16):
        """
        Args:
            dataset_info: Dataset information
            compounds_dir: Path to the compounds directory
            num_processes: Number of parallel processing processes
        """
        self.dataset_info = dataset_info
        self.compounds_dir = Path(compounds_dir)
        self.num_processes = num_processes

    def process(self, force: bool = False) -> Optional[pd.DataFrame]:
        """
        Process the dataset

        Args:
            force: Force reprocessing flag

        Returns:
            Processed DataFrame (None on error)
        """
        processed_path = self.dataset_info.get_processed_path(self.compounds_dir)

        # Skip if already processed
        if not force and processed_path.exists():
            logger.info(f"✓ {self.dataset_info.name}: Already processed, skipping")
            return pd.read_parquet(processed_path)

        # read raw data
        df = self._load_raw_data()
        if df is None:
            return None

        # sampling
        if self.dataset_info.sample_size is not None and len(df) > self.dataset_info.sample_size:
            logger.info(f"  Sampling {self.dataset_info.sample_size} from {len(df)} samples")
            df = df.sample(n=self.dataset_info.sample_size, random_state=42)

        # Physical property calculation
        if self.dataset_info.requires_properties:
            df = self._calculate_properties(df)
            if df is None:
                return None

        # keep
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path, index=False)
        logger.info(f"✓ {self.dataset_info.name}: Saved {len(df)} samples to {processed_path}")

        return df

    def _load_raw_data(self) -> Optional[pd.DataFrame]:
        """Load raw data"""
        raw_path = self.dataset_info.get_raw_path(self.compounds_dir)

        if not raw_path.exists():
            logger.warning(
                f"⚠ {self.dataset_info.name}: Raw data not found at {raw_path}\n  Please download this dataset first."
            )
            return None

        try:
            logger.info(f"📂 {self.dataset_info.name}: Loading from {raw_path}")
            df = pd.read_parquet(raw_path)

            # check that SMILES column exists
            if "smiles" not in df.columns:
                logger.error(f"✗ {self.dataset_info.name}: 'smiles' column not found")
                return None

            logger.info(f"✓ {self.dataset_info.name}: Loaded {len(df)} samples")
            return df

        except Exception as e:
            logger.error(f"✗ {self.dataset_info.name}: Failed to load - {e}")
            return None

    def _calculate_properties(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate physical properties"""
        logger.info(f"🧪 {self.dataset_info.name}: Calculating properties...")

        try:
            smi_series = df["smiles"]

            with multiprocessing.Pool(self.num_processes) as pool:
                # LogPcalculation
                logger.info("  Computing LogP...")
                logps_list = pool.map(calcLogPIfMol, smi_series)

                # filter only valid molecules
                valid_mols = ~pd.isna(logps_list)
                valid_smiles = smi_series[valid_mols].reset_index(drop=True)
                valid_logps = pd.Series(logps_list)[valid_mols].reset_index(drop=True)

                # Molecular weight calculation
                logger.info("  Computing molecular weight...")
                mol_weights = pool.map(calcMolWeight, valid_smiles)

                # SA scorecalculation
                logger.info("  Computing SA score...")
                sascores = pool.map(calcSascore, valid_smiles)

            # Convert result to DataFrame
            result_df = pd.DataFrame(
                {
                    "smiles": valid_smiles,
                    "logp": valid_logps,
                    "mol_weight": [w / 100.0 if w is not None else None for w in mol_weights],  # Normalization
                    "sascore": sascores,
                }
            )

            # Delete lines containing NaN
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
    Process all available datasets

    Args:
        compounds_dir: compounds directorypath of
        dataset_types: List of dataset types to process (all available if None)
        force: Force reprocessing flag
        num_processes: Number of parallel processing processes

    Returns:
        Dictionary of {dataset_type: processed_df}
    """
    from molcrawl.data.compounds.dataset.dataset_config import (
        get_available_datasets,
        get_dataset_info,
    )

    # Determine the dataset to be processed
    if dataset_types is None:
        # Get all available datasets
        available = get_available_datasets(compounds_dir)
        if not available:
            logger.warning("No datasets available for processing")
            return {}
        dataset_types = available
    else:
        # If the specified dataset is a string, convert it to an Enum
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
