from src.compounds.utils.datasets import download


DATASETS_FROM_REPO = [
    {
        "url": "https://github.com/Fraunhofer-SCAI/llamol/raw/f6d98e6fee5bf26aa7777cdbb3a55d518d13eeaf/data/qm9_zinc250k_cep/qm9_zinc250_cep.parquet",
        "name": "qm9_zinc250_cep.parquet"
    },
    {
        "url": "https://github.com/Fraunhofer-SCAI/llamol/raw/f6d98e6fee5bf26aa7777cdbb3a55d518d13eeaf/data/Full_PC9_GAP.parquet",
        "name": "Full_PC9_GAP.parquet"
    },
    {
        "url": "https://github.com/Fraunhofer-SCAI/llamol/raw/f6d98e6fee5bf26aa7777cdbb3a55d518d13eeaf/data/RedDB_Full.parquet",
        "name": "RedDB_Full.parquet"
    },
    {
        "url": "https://github.com/Fraunhofer-SCAI/llamol/raw/f6d98e6fee5bf26aa7777cdbb3a55d518d13eeaf/data/chembl_log_sascore.parquet",
        "name": "chembl_log_sascore.parquet"
    },
    {
        "url": "https://github.com/Fraunhofer-SCAI/llamol/raw/f6d98e6fee5bf26aa7777cdbb3a55d518d13eeaf/data/pubchemqc2020_energy.parquet",
        "name": "pubchemqc2020_energy.parquet"
    },
    {
        "url": "https://github.com/Fraunhofer-SCAI/llamol/raw/f6d98e6fee5bf26aa7777cdbb3a55d518d13eeaf/data/pubchemqc_energy.parquet",
        "name": "pubchemqc_energy.parquet"
    }
]


def download_datasets_from_repo(output_dir: str):
    for dataset in DATASETS_FROM_REPO:
        download(output_dir, **dataset)
