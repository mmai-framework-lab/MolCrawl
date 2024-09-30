from typing import Tuple, List
import os
import joblib
from pathlib import Path
import logging
import concurrent.futures
import time
from functools import partial

import cellxgene_census
import numpy as np
import pandas as pd

# from tqdm import tqdm
from rich.progress import track
import anndata
import tiledbsoma as soma

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def retrieve_census(try_count: int = 0, max_try: int = 5) -> soma.Collection:
    try:
        return cellxgene_census.open_soma(census_version="2023-12-15")
    except KeyboardInterrupt:
        return
    except Exception as e:
        if try_count > max_try:
            raise e
        logging.warning(
            f"[Error] while retrieving census, retrying (try: {try_count+1})"
        )
        time.sleep(10)
        return retrieve_census(try_count + 1)


def retrieve_adata(
    id_list: List[int],
    target_gene_ids: np.ndarray,
    try_count: int = 0,
    max_try: int = 5,
) -> anndata.AnnData:
    census = retrieve_census()
    try:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_coords=id_list,
            var_coords=target_gene_ids,
        )
    except KeyboardInterrupt:
        return
    except Exception as e:
        if try_count > max_try:
            census.close()
            raise e
        logging.warning(
            f"[Error] while retrieving adata, retrying (try: {try_count+1})"
        )
        time.sleep(10)
        adata = retrieve_adata(id_list, target_gene_ids, try_count + 1)

    census.close()
    return adata


def run(output_dir: Path, argv: Tuple[str, int, int, List[int]]) -> None:
    name, start_l, end_l, id_list = argv
    save_filename = output_dir / f"01data/{name}.{start_l:08d}-{end_l:08d}.jbl"
    if save_filename.exists():
        logging.info(f"{save_filename} exists, skipping download")
        return

    target_var = pd.read_csv(
        output_dir / "00data" / f"{name}.var.tsv", sep="\t", index_col=0
    )
    target_gene_ids = target_var["soma_joinid"].to_numpy()

    target_adata = retrieve_adata(id_list, target_gene_ids)

    joblib.dump(target_adata, save_filename, compress=3)
    time.sleep(1)


def divide_workload(
    path: str, size_workload: int
) -> List[Tuple[str, int, int, List[int]]]:
    divided_workload = []
    for filename in Path(path).rglob("*.obs_id.tsv"):
        with open(filename, "r") as file:
            id_list = np.array(file.readlines()).astype(int)
        name = Path(filename).stem.split(".")[0]
        start_lines = range(0, len(id_list), size_workload)
        end_lines = list(range(size_workload, len(id_list), size_workload)) + [
            len(id_list)
        ]

        divided_workload += [
            (name, start, end, id_list[start:end].tolist())
            for start, end in zip(start_lines, end_lines)
        ]

    return divided_workload


def download(output_dir, num_worker, size_workload):
    path_data_directory = Path(output_dir) / "00data"
    path_download_directory = Path(output_dir) / "01data"

    os.makedirs(path_download_directory, exist_ok=True)
    arg_list = divide_workload(path_data_directory, size_workload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
        func = partial(run, Path(output_dir))
        list(
            track(
                executor.map(func, arg_list),
                description="Downloading...",
                total=len(arg_list),
            )
        )


if __name__ == "__main__":
    size_workload = 10000
    num_worker = 8
    output_dir = "CellxgenePreprocessor"

    download(output_dir, num_worker, size_workload)
