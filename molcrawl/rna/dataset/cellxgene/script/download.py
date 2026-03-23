import concurrent.futures
import logging
import os
import socket
import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

from rich.progress import track

from molcrawl.rna.utils.config import RnaConfig

# Prevent CellxGene API calls from hanging indefinitely.
# Each socket operation (connect, read) will raise socket.timeout after this.
_SOCKET_TIMEOUT_SEC = 300  # 5 minutes
# Wall-clock timeout for the entire download+write of one chunk.
# Needed because socket.setdefaulttimeout only bounds individual recv() calls,
# not the total transfer time when the API trickles data slowly.
_DOWNLOAD_TOTAL_TIMEOUT_SEC = 600  # 10 minutes per attempt
_DOWNLOAD_MAX_RETRY = 5

# Prevent CellxGene API calls from hanging indefinitely.
# Each socket operation (connect, read) will raise socket.timeout after this.
_SOCKET_TIMEOUT_SEC = 300  # 5 minutes
# Wall-clock timeout for the entire download+write of one chunk.
# Needed because socket.setdefaulttimeout only bounds individual recv() calls,
# not the total transfer time when the API trickles data slowly.
_DOWNLOAD_TOTAL_TIMEOUT_SEC = 1200  # 20 minutes per attempt
_DOWNLOAD_MAX_RETRY = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def retrieve_census(version: str, try_count: int = 0, max_try: int = 5) -> Any:
    import cellxgene_census
    import tiledbsoma

    # Access S3 directly instead of via the SOMA HTTPS API endpoint.
    # This bypasses the API server bottleneck and reads TileDB chunks from S3
    # client-side, which is significantly faster for bulk downloads.
    ctx = tiledbsoma.SOMATileDBContext(
        tiledb_config={
            "vfs.s3.region": "us-west-2",
            "vfs.s3.no_sign_request": "true",
        }
    )
    try:
        return cellxgene_census.open_soma(census_version=version, context=ctx)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        if try_count > max_try:
            raise e
        logging.warning(f"[Error] while retrieving census, retrying (try: {try_count + 1})")
        time.sleep(10)
        return retrieve_census(version, try_count + 1)


def retrieve_adata(
    version: str,
    id_list: List[int],
    target_gene_ids: Sequence[int],
    try_count: int = 0,
    max_try: int = 5,
) -> Any:
    import cellxgene_census

    census = retrieve_census(version)
    try:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_coords=id_list,
            var_coords=target_gene_ids,
        )
    except KeyboardInterrupt as e:
        census.close()
        raise e
    except Exception as e:
        census.close()
        if try_count > max_try:
            raise e
        logging.warning(f"[Error] while retrieving adata, retrying (try: {try_count + 1})")
        time.sleep(10)
        return retrieve_adata(version, id_list, target_gene_ids, try_count + 1)
    else:
        census.close()
    return adata


def _do_download_and_write(
    save_filename: Path,
    version: str,
    id_list: List[int],
    target_gene_ids: Sequence[int],
) -> None:
    """Download one chunk from CellxGene and write to h5ad. Runs in a helper thread."""
    socket.setdefaulttimeout(_SOCKET_TIMEOUT_SEC)
    target_adata = retrieve_adata(version, id_list, target_gene_ids)
    target_adata.write_h5ad(save_filename, compression="gzip")


def run(output_dir: Path, version, argv: Tuple[str, int, int, List[int]]) -> None:
    import pandas as pd

    name, start_l, end_l, id_list = argv
    save_filename = output_dir / f"download_dir/{name}.{start_l:08d}-{end_l:08d}.h5ad"
    if save_filename.exists():
        try:
            import h5py

            with h5py.File(save_filename, "r") as _f:
                pass  # header check only — avoids loading full data into memory
            logging.info(f"{save_filename} exists, skipping download")
            return
        except Exception as e:
            logging.warning(f"{save_filename} is corrupt ({e}), re-downloading")
            save_filename.unlink()

    tsv_file = output_dir / "metadata_preparation_dir" / f"{name}.var.tsv"
    target_var = pd.read_csv(tsv_file, sep="\t", index_col=0)
    target_gene_ids = target_var["soma_joinid"].to_numpy()

    for attempt in range(1, _DOWNLOAD_MAX_RETRY + 1):
        # Each attempt runs in a dedicated thread so we can apply a hard wall-clock
        # timeout via future.result(timeout=…).  The abandoned thread will terminate
        # on its own once _SOCKET_TIMEOUT_SEC fires on the next socket recv().
        _exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = _exec.submit(_do_download_and_write, save_filename, version, id_list, target_gene_ids)
        timed_out = failed = False
        try:
            fut.result(timeout=_DOWNLOAD_TOTAL_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            timed_out = True
            logging.warning(
                f"[Timeout] Download exceeded {_DOWNLOAD_TOTAL_TIMEOUT_SEC}s wall-clock "
                f"(attempt {attempt}/{_DOWNLOAD_MAX_RETRY}): {save_filename}"
            )
        except Exception as exc:
            failed = True
            logging.warning(f"[Error] Download failed: {exc} (attempt {attempt}/{_DOWNLOAD_MAX_RETRY}): {save_filename}")
        finally:
            _exec.shutdown(wait=False)  # don't block; stuck thread ends via socket timeout

        if not timed_out and not failed:
            logging.info(f"Downloaded {save_filename} ({len(id_list)} cells)")
            return  # success

        # Remove any partial file before retrying
        if save_filename.exists():
            try:
                save_filename.unlink()
            except OSError:
                pass

        if attempt < _DOWNLOAD_MAX_RETRY:
            time.sleep(10)

    # All retries exhausted — log and skip so the worker continues with remaining chunks
    # instead of crashing the entire ThreadPoolExecutor.
    logging.error(f"Skipping {save_filename}: all {_DOWNLOAD_MAX_RETRY} download attempts failed")


def divide_workload(path: Union[str, Path], size_workload: int) -> List[Tuple[str, int, int, List[int]]]:
    """
    Function to split workload into specified sizes

    Read the *.obs_id.tsv file in the specified directory,
    Split each file's ID list into specified workload sizes.

        Args:
    path (Union[str, Path]): path of metadata preparation directory
    size_workload (int): Size of each workload (number of samples)

        Returns:
    List[Tuple[str, int, int, List[int]]]: List of partitioned workloads
    - str: file name (no extension)
    - int: starting line number
    - int: end line number
    - List[int]: ID list of applicable range
    """
    divided_workload = []
    for filename in Path(path).rglob("*.obs_id.tsv"):
        with open(filename, "r") as file:
            import numpy as np

            id_list = np.array(file.readlines()).astype(int)
        name = Path(filename).stem.split(".")[0]
        start_lines = range(0, len(id_list), size_workload)
        end_lines = list(range(size_workload, len(id_list), size_workload)) + [len(id_list)]

        divided_workload += [(name, start, end, id_list[start:end].tolist()) for start, end in zip(start_lines, end_lines)]

    return divided_workload


def download(output_dir, version, num_worker, size_workload):
    path_data_directory = Path(output_dir) / "metadata_preparation_dir"
    path_download_directory = Path(output_dir) / "download_dir"

    os.makedirs(path_download_directory, exist_ok=True)
    arg_list = divide_workload(path_data_directory, size_workload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
        func = partial(run, Path(output_dir), version)
        list(
            track(
                executor.map(func, arg_list),
                description="Downloading...",
                total=len(arg_list),
            )
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    download(cfg.output_dir, cfg.census_version, cfg.num_worker, cfg.size_workload)
