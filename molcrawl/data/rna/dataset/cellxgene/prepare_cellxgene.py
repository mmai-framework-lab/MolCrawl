"""
This script will download the cellxgene dataset.
There will be multiple directory generate in the output_dir provided in the configuration

- download_dir: Raw archive file downloaded from the cellxgene database
- extract: h5ad file extracted from the archives
- parquet_files: parquet files containing tokenized gene and expression values
"""

from argparse import ArgumentParser

from molcrawl.data.rna.utils.config import RnaConfig


if __name__ == "__main__":
    from molcrawl.data.rna.dataset.cellxgene.script.build_list import build_list
    from molcrawl.data.rna.dataset.cellxgene.script.download import download
    from molcrawl.data.rna.dataset.cellxgene.script.h5ad_to_loom import h5ad_to_loom

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    build_list(cfg.output_dir, cfg.census_version)
    download(cfg.output_dir, cfg.census_version, cfg.num_worker, cfg.size_workload)
    h5ad_to_loom(cfg.output_dir)
