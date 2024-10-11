from typing import Union
import gc
from pathlib import Path
import traceback
from typing import Dict, List, Optional
import warnings
import os
import concurrent.futures
import logging
from functools import partial
from argparse import ArgumentParser


import rich.progress
import rich.progress_bar
import scanpy as sc
from scgpt import scbank
import scgpt as scg
import cellxgene_census
import rich
from datasets.utils.logging import disable_progress_bar

from rna.utils.config import RnaConfig

disable_progress_bar()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess(
    adata: sc.AnnData,
    main_table_key: str = "counts",
    include_obs: Optional[Dict[str, List[str]]] = None,
    min_counts_genes: int = 2,
) -> sc.AnnData:
    """
    Preprocess the data for scBank. This function will modify the AnnData object in place.

    Args:
        adata: AnnData object to preprocess
        main_table_key: key in adata.layers to store the main table
        include_obs: dict of column names and values to include in the main table

    Returns:
        The preprocessed AnnData object
    """
    if include_obs is not None:
        # include only cells that have the specified values in the specified columns
        for col, values in include_obs.items():
            adata = adata[adata.obs[col].isin(values)]

    # filter genes
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)

    # TODO: add binning in sparse matrix and save in separate datatable
    # The binning happens in the Collator
    # preprocessor = Preprocessor(
    #     use_key="X",  # the key in adata.layers to use as raw data
    #     filter_gene_by_counts=False,  # step 1
    #     filter_cell_by_counts=False,  # step 2
    #     normalize_total=False,  # 3. whether to normalize the raw data and to what sum
    #     log1p=False,  # 4. whether to log1p the normalized data
    #     binning=51,  # 6. whether to bin the raw data and to what number of bins
    #     result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    # )
    # preprocessor(adata)

    adata.layers[main_table_key] = adata.X.copy()

    return adata


def process_h5ad_to_parquet(
    h5ad_path: Union[str, Path], output_dir: Union[str, Path], vocab: scg.tokenizer.GeneVocab, min_counts_genes: int
):
    h5ad_path, output_dir = Path(h5ad_path), Path(output_dir)
    parquet_path = output_dir / h5ad_path.with_suffix(".parquet").name
    if not parquet_path.exists():
        try:
            adata = sc.read(h5ad_path, cache=True)
            adata = preprocess(adata, min_counts_genes=min_counts_genes)
            print(f"read {adata.shape} valid data from {h5ad_path.name}")

            db = scbank.DataBank.from_anndata(
                adata,
                vocab=vocab,
                to=output_dir,
                main_table_key="counts",
                token_col="feature_name",
                immediate_save=False,
            )
            db.data_tables["counts"].data.to_parquet(parquet_path)

            # clean up
            del adata
            del db
            gc.collect()
        except Exception as e:
            traceback.print_exc()
            warnings.warn(f"failed to process {h5ad_path}: {e}")
            if parquet_path.exists():
                os.remove(parquet_path)
    else:
        logging.info(f"Skipping processing since file alreafy exist: {parquet_path}")


def get_census_gene_vocab(version: str):
    with cellxgene_census.open_soma(census_version=version) as census:
        meta_data = (
            census["census_data"]["homo_sapiens"]
            .ms["RNA"]
            .var.read(
                column_names=[
                    "feature_name",
                ],
            )
        )

    gene_list = meta_data.concat().to_pandas()["feature_name"].to_list()
    return scg.tokenizer.GeneVocab(gene_list)


def prepare_parquet(output_dir: str, version: str, num_worker: int, min_counts_genes: int):

    vocab = get_census_gene_vocab(version)
    vocab.save_json(Path(output_dir) / "gene_vocab.json")

    input_dir = Path(output_dir) / "download_dir"

    files = list(Path(input_dir).glob("*.h5ad"))
    parquet_dir = Path(output_dir) / "parquet_files"
    parquet_dir.mkdir(exist_ok=True, parents=True)

    with concurrent.futures.ThreadPoolExecutor(num_worker) as executor:
        func = partial(process_h5ad_to_parquet, output_dir=parquet_dir, vocab=vocab, min_counts_genes=min_counts_genes)
        list(rich.progress.track(executor.map(func, files), "Tokenizing h5ad file to parquet...", len(files)))


if "__main__" == __name__:
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    prepare_parquet(cfg.output_dir, cfg.census_version, cfg.num_worker, cfg.min_counts_genes)
