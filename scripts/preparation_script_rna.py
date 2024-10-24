"""
This script will download the cellxgene dataset.
There will be multiple directory generate in the output_dir provided in the configuration

- download_dir: Raw archive file downloaded from the cellxgene database
- extract: h5ad file extracted from the archives
- parquet_files: parquet files containing tokenized gene and expression values

You can call this script with the following command:

python scripts/preparation_script_rna.py assets/configs/rna.yaml
"""

from argparse import ArgumentParser

from rna.dataset.cellxgene.script.build_list import build_list
from rna.dataset.cellxgene.script.download import download
from rna.dataset.cellxgene.script.tokenization import prepare_parquet
from rna.utils.config import RnaConfig


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    build_list(cfg.output_dir)
    download(cfg.output_dir, cfg.census_version, cfg.num_worker, cfg.size_workload)
    prepare_parquet(cfg.output_dir, cfg.census_version, cfg.num_worker, cfg.min_counts_genes)
