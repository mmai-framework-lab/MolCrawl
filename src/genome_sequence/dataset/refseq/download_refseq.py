from typing import List, Union
import re
import os
import gzip
import json
import requests
from urllib.request import urlretrieve
import shutil
import logging
import logging.config
from pathlib import Path
import concurrent.futures
from functools import partial
import traceback
from argparse import ArgumentParser

from rich.progress import track

from genome_sequence.utils.config import GenomeSequenceConfig

logger = logging.getLogger(__name__)


def get_list_files(url: str) -> List[str]:
    response = requests.get(url)
    response.raise_for_status()
    # Use a regex pattern to find all href links in the page
    files = re.findall(r'href="([^"]+).genomic.fna.gz"', response.text)

    return [f"{file}.fna.gz" for file in files]


def download(url: str, path: str, try_count: int = 0, max_try: int = 3) -> str:
    """
    Download a file from the specified url.
    Skip the downloading step if there exists a file with the same name

    Parameters:
        url (str): URL to download
        path (str, optional): path to store the downloaded file. If not specify tmp file.
    """

    if not os.path.exists(path):
        logger.info("Downloading %s to %s" % (url, path))
        try:
            path, _ = urlretrieve(url, path)
        except Exception as e:
            os.remove(path)
            msg = str(e) + "\n" + "".join(traceback.format_exception(None, e, e.__traceback__))
            logger.error(f"[Try: {try_count+1}] Error while downloading {path}: \n{msg}")
            if try_count < max_try:
                return download(url, path, try_count + 1)
    else:
        logger.info("Skipping %s since already downloaded at %s" % (url, path))

    return path


def extract_file(
    archive_path: str,
    output_dir: os.PathLike[str],
    try_count: int = 0,
    max_try: int = 3,
):
    # pass .sdf.gz to .sdf
    sdf_file_path = Path(output_dir) / Path(archive_path).with_suffix("").name
    if sdf_file_path.exists():
        logger.info(f"Skipping extraction of {sdf_file_path}, already exist")
        return
    logger.info(f"Extracting {archive_path} to {sdf_file_path}")
    if os.path.exists(archive_path):
        try:
            # Decompress the .gz file and save the result as .sdf
            with gzip.open(archive_path, "rb") as f_in:
                with open(sdf_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            os.remove(sdf_file_path)
            msg = str(e) + "\n" + "".join(traceback.format_exception(None, e, e.__traceback__))
            logging.error(f"[Try: {try_count+1}]  File {archive_path} created an error : \n{msg}")
            if try_count < max_try:
                return extract_file(archive_path, output_dir, try_count + 1)
    else:
        logger.error(f"File {archive_path} does not exist skipping")


def download_refseq(output_dir: Union[str, os.PathLike[str]], num_worker: int):
    base_url = "https://ftp.ncbi.nlm.nih.gov/refseq/release/complete/"
    logger.info(f"Downloading PubMed from {base_url}")

    output_dir = Path(output_dir)
    download_dir = output_dir / "download_dir"
    extracted_dir = output_dir / "extracted_dir"

    download_dir.mkdir(parents=True, exist_ok=True)
    files = get_list_files(base_url)
    urls = [os.path.join(base_url, file) for file in files]
    paths = [os.path.join(download_dir, file) for file in files]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_worker, 3)) as executor:
        archive_paths = list(
            track(
                executor.map(download, urls, paths),
                total=len(urls),
                description="Downloading...",
            )
        )

    extracted_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
        func = partial(extract_file, output_dir=extracted_dir)
        list(
            track(
                executor.map(func, archive_paths),
                total=len(archive_paths),
                description="Extracting...",
            )
        )


def setup_logging(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open("./assets/logging-config.json", "r") as file:
        config = json.load(file)
    logging_file = f"{output_dir}/logging.log"
    config["handlers"]["file"]["filename"] = logging_file
    if os.path.exists(logging_file):
        os.remove(logging_file)
    logging.config.dictConfig(config=config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    setup_logging(cfg.output_dir)
    download_refseq(cfg.output_dir, cfg.num_worker)
