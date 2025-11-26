import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import multiprocessing

from rdkit.Chem import RDConfig
import sys

import logging

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
# now you can import sascore!
import sascorer  # noqa: E402

np.random.seed(42)


def calcLogPIfMol(smi):
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return Descriptors.MolLogP(m)
    else:
        return None


def calcMol(smi):
    return Chem.MolFromSmiles(smi)


def calcMolWeight(smi):
    mol = Chem.MolFromSmiles(smi)
    return Descriptors.ExactMolWt(mol)


def calcSascore(smi):
    mol = Chem.MolFromSmiles(smi)

    return sascorer.calculateScore(mol)


def calculateValues(smi: pd.Series):

    logging.info("Calculating properties")
    with multiprocessing.Pool(16) as pool:
        logging.info("Starting logps")
        logps = pool.map(calcLogPIfMol, smi)

        valid_mols = ~pd.isna(logps)
        logps = pd.Series(logps)[valid_mols]
        smi = pd.Series(smi)[valid_mols]
        logps.reset_index(drop=True, inplace=True)
        smi.reset_index(drop=True, inplace=True)
        logging.info("Starting mol weights")

        mol_weights = pool.map(calcMolWeight, smi)

        logging.info("Starting sascores")

        sascores = pool.map(calcSascore, smi)


    return smi, logps, mol_weights, sascores


def calculateProperties(df):

    smi, logps, mol_weights, sascores = calculateValues(df["smiles"])
    out_df = pd.DataFrame(
        {"smiles": smi, "logp": logps, "mol_weight": mol_weights, "sascore": sascores}
    )

    return out_df



def combine_all(raw_data_path: str, save_path: str):
    """
    全データセットを統合してOrganiX13を生成
    
    Args:
        raw_data_path: COMPOUNDS_DIR (例: learning_20251104/compounds)
        save_path: 出力先ディレクトリ (例: learning_20251104/compounds/organix13)
    """
    # データディレクトリのパス
    data_dir = os.path.join(raw_data_path, "data")
    llamol_dir = os.path.join(data_dir, "Fraunhofer-SCAI-llamol")
    
    logging.info("Processing df_pc9")
    df_pc9 = pd.read_parquet(os.path.join(llamol_dir, "Full_PC9_GAP.parquet"))
    df_pc9 = calculateProperties(df_pc9)

    logging.info("Processing df_zinc_full")
    df_zinc_full = pd.read_parquet(os.path.join(data_dir, "zinc20", "zinc_processed.parquet"))
    df_zinc_full = df_zinc_full.sample(n=5_000_000)
    df_zinc_full = calculateProperties(df_zinc_full)

    logging.info("Processing df_zinc_qm9")
    df_zinc_qm9 = pd.read_parquet(
        os.path.join(llamol_dir, "qm9_zinc250_cep.parquet")
    )
    df_zinc_qm9 = calculateProperties(df_zinc_qm9)

    logging.info("Processing df_opv")
    df_opv = pd.read_parquet(os.path.join(data_dir, "opv", "opv.parquet"))
    df_opv = calculateProperties(df_opv)

    logging.info("Processing df_reddb")
    # Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/F3QFSQ
    df_reddb = pd.read_parquet(os.path.join(llamol_dir, "RedDB_Full.parquet"))
    df_reddb = calculateProperties(df_reddb)

    logging.info("Processing df_chembl")
    df_chembl = pd.read_parquet(os.path.join(llamol_dir, "chembl_log_sascore.parquet"))
    df_chembl = calculateProperties(df_chembl)

    logging.info("Processing df_pubchemqc_2017")
    df_pubchemqc_2017 = pd.read_parquet(os.path.join(llamol_dir, "pubchemqc_energy.parquet"))
    df_pubchemqc_2017 = calculateProperties(df_pubchemqc_2017)

    logging.info("Processing df_pubchemqc_2020")
    df_pubchemqc_2020 = pd.read_parquet(
        os.path.join(llamol_dir, "pubchemqc2020_energy.parquet")
    )
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

    df = pd.concat(
        df_list, axis=0, ignore_index=True
    ) 
    df = df[all_columns]
    df.reset_index(drop=True, inplace=True)
    df["mol_weight"] = df["mol_weight"] / 100.0

    logging.info(df.head())
    logging.info("saving")
    logging.info("Combined len: {}".format(len(df)))
    df.to_parquet(os.path.join(save_path, "OrganiX13.parquet"))
