"""
Legacy code use for scgpt, in the end geneformer is used to compute the tokenization
"""

from typing import Any, TYPE_CHECKING, Union
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


if TYPE_CHECKING:
    import scanpy as sc
    import scgpt as scg
    from scgpt import scbank

from rna.utils.config import RnaConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess(
    adata: Any,
    main_table_key: str = "counts",
    include_obs: Optional[Dict[str, List[str]]] = None,
    min_counts_genes: int = 2,
) -> Any:
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
    import scanpy as sc

    sc.pp.filter_genes(adata, min_counts=min_counts_genes)
    adata.layers[main_table_key] = adata.X.copy()

    return adata


def process_h5ad_to_parquet(
    h5ad_path: Union[str, Path], output_dir: Union[str, Path], vocab: Any, min_counts_genes: int
):
    h5ad_path, output_dir = Path(h5ad_path), Path(output_dir)
    parquet_path = output_dir / h5ad_path.with_suffix(".parquet").name
    if not parquet_path.exists():
        try:
            import scanpy as sc
            from scgpt import scbank

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
            data = db.data_tables["counts"].data
            data = data.map(lambda expression: {"num_tokens": len(expression["genes"])})
            data.to_parquet(parquet_path)
            logging.info(f"Saving file to {parquet_path}")

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
        logging.info(f"Skipping processing since file already exist: {parquet_path}")


def get_census_gene_vocab(version: str):
    import cellxgene_census
    import scgpt as scg

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
    import rich.progress
    from datasets.utils.logging import disable_progress_bar, enable_progress_bar

    disable_progress_bar()

    vocab = get_census_gene_vocab(version)
    vocab.save_json(Path(output_dir) / "gene_vocab.json")

    input_dir = Path(output_dir) / "download_dir"

    files = list(Path(input_dir).glob("*.h5ad"))
    parquet_dir = Path(output_dir) / "parquet_files"
    parquet_dir.mkdir(exist_ok=True, parents=True)

    with concurrent.futures.ThreadPoolExecutor(num_worker) as executor:
        func = partial(process_h5ad_to_parquet, output_dir=parquet_dir, vocab=vocab, min_counts_genes=min_counts_genes)
        list(rich.progress.track(executor.map(func, files), "Tokenizing h5ad file to parquet...", len(files)))

    enable_progress_bar()


if "__main__" == __name__:
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    prepare_parquet(cfg.output_dir, cfg.census_version, cfg.num_worker, cfg.min_counts_genes)
