from __future__ import annotations

import numpy as np
import os
import multiprocessing
from typing import TYPE_CHECKING

import logging

logger = logging.getLogger(__name__)

np.random.seed(42)

if TYPE_CHECKING:
    import pandas as pd


def _get_rdkit_helpers():
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    import sascorer

    return Chem, Descriptors, sascorer


def safe_read_parquet(file_path, dataset_name):
    import pandas as pd

    """
    Safely read a parquet file with error handling

    Args:
        file_path: Path to the parquet file
        dataset_name: Name of the dataset for logging

    Returns:
        DataFrame or None if file is corrupted

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is corrupted
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{dataset_name} parquet file not found: {file_path}\nPlease re-run the download step to obtain this file."
        )

    try:
        logger.info(f"Reading {dataset_name} from {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded {dataset_name}: {len(df)} rows")
        return df
    except Exception as e:
        error_msg = (
            f"Failed to read {dataset_name} parquet file: {file_path}\n"
            f"Error: {str(e)}\n"
            f"The file may be corrupted or incomplete.\n"
            f"Solution: Delete the file and re-run the download:\n"
            f"  rm {file_path}\n"
            f"  LEARNING_SOURCE_DIR=<your_dir> ./bootstraps/01_compounds_prepare.sh"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def calcLogPIfMol(smi):
    Chem, Descriptors, _ = _get_rdkit_helpers()
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return Descriptors.MolLogP(m)
    else:
        return None


def calcMol(smi):
    Chem, _, _ = _get_rdkit_helpers()
    return Chem.MolFromSmiles(smi)


def calcMolWeight(smi):
    Chem, Descriptors, _ = _get_rdkit_helpers()
    mol = Chem.MolFromSmiles(smi)
    return Descriptors.ExactMolWt(mol)


def calcSascore(smi):
    Chem, _, sascorer = _get_rdkit_helpers()
    mol = Chem.MolFromSmiles(smi)

    return sascorer.calculateScore(mol)


def calculateValues(smi: pd.Series):
    import pandas as pd

    logging.info("Calculating properties")
    with multiprocessing.Pool(16) as pool:
        logging.info("Starting logps")
        logps_list = pool.map(calcLogPIfMol, smi)

        valid_mols = ~pd.isna(logps_list)
        logps = pd.Series(logps_list)[valid_mols]
        smi = pd.Series(smi)[valid_mols]
        logps.reset_index(drop=True, inplace=True)
        smi.reset_index(drop=True, inplace=True)
        logging.info("Starting mol weights")

        mol_weights = pool.map(calcMolWeight, smi)

        logging.info("Starting sascores")

        sascores = pool.map(calcSascore, smi)

    return smi, logps, mol_weights, sascores


def calculateProperties(df):
    import pandas as pd

    smi, logps, mol_weights, sascores = calculateValues(df["smiles"])
    out_df = pd.DataFrame({"smiles": smi, "logp": logps, "mol_weight": mol_weights, "sascore": sascores})

    return out_df


def combine_all(raw_data_path: str, save_path: str):
    """
    全データセットを統合してOrganiX13を生成

    Args:
        raw_data_path: COMPOUNDS_DIR (例: learning_20251104/compounds)
        save_path: 出力先ディレクトリ (例: learning_20251104/compounds/organix13)
    """
    import pandas as pd

    # データディレクトリのパス
    data_dir = os.path.join(raw_data_path, "data")
    llamol_dir = os.path.join(data_dir, "Fraunhofer-SCAI-llamol")

    logging.info("Processing df_pc9")
    df_pc9 = safe_read_parquet(os.path.join(llamol_dir, "Full_PC9_GAP.parquet"), "PC9 GAP")
    df_pc9 = calculateProperties(df_pc9)

    logging.info("Processing df_zinc_full")
    df_zinc_full = safe_read_parquet(os.path.join(data_dir, "zinc20", "zinc_processed.parquet"), "ZINC20 Full")
    df_zinc_full = df_zinc_full.sample(n=5_000_000)
    df_zinc_full = calculateProperties(df_zinc_full)

    logging.info("Processing df_zinc_qm9")
    df_zinc_qm9 = safe_read_parquet(os.path.join(llamol_dir, "qm9_zinc250_cep.parquet"), "ZINC QM9")
    df_zinc_qm9 = calculateProperties(df_zinc_qm9)

    logging.info("Processing df_opv")
    df_opv = safe_read_parquet(os.path.join(data_dir, "opv", "opv.parquet"), "OPV")
    df_opv = calculateProperties(df_opv)

    logging.info("Processing df_reddb")
    # Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/F3QFSQ
    df_reddb = safe_read_parquet(os.path.join(llamol_dir, "RedDB_Full.parquet"), "RedDB")
    df_reddb = calculateProperties(df_reddb)

    logging.info("Processing df_chembl")
    df_chembl = safe_read_parquet(os.path.join(llamol_dir, "chembl_log_sascore.parquet"), "ChEMBL")
    df_chembl = calculateProperties(df_chembl)

    logging.info("Processing df_pubchemqc_2017")
    df_pubchemqc_2017 = safe_read_parquet(os.path.join(llamol_dir, "pubchemqc_energy.parquet"), "PubChemQC 2017")
    df_pubchemqc_2017 = calculateProperties(df_pubchemqc_2017)

    logging.info("Processing df_pubchemqc_2020")
    df_pubchemqc_2020 = safe_read_parquet(os.path.join(llamol_dir, "pubchemqc2020_energy.parquet"), "PubChemQC 2020")
    df_pubchemqc_2020 = calculateProperties(df_pubchemqc_2020)

    df_list = [
        df_zinc_qm9,
        df_opv,
        df_pubchemqc_2017,
        df_pubchemqc_2020,
        df_zinc_full,
        df_reddb,
        df_pc9,
        df_chembl,
    ]

    logging.info(f"ZINC QM9 {len(df_zinc_qm9)}")
    logging.info(f"df_opv {len(df_opv)}")
    logging.info(f"df_pubchemqc_2017 {len(df_pubchemqc_2017)}")
    logging.info(f"df_pubchemqc_2020 {len(df_pubchemqc_2020)}")
    logging.info(f"df_zinc_full {len(df_zinc_full)}")
    logging.info(f"df_reddb {len(df_reddb)}")
    logging.info(f"df_pc9 {len(df_pc9)}")
    logging.info(f"df_chembl {len(df_chembl)}")

    all_columns = [
        "smiles",
        "logp",
        "sascore",
        "mol_weight",
    ]

    logging.info("concatenting")

    df = pd.concat(df_list, axis=0, ignore_index=True)
    df = df[all_columns]
    df.reset_index(drop=True, inplace=True)
    df["mol_weight"] = df["mol_weight"] / 100.0

    logging.info(df.head())
    logging.info("saving")
    logging.info("Combined len: {}".format(len(df)))
    df.to_parquet(os.path.join(save_path, "OrganiX13.parquet"))
