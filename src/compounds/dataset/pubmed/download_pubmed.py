"""
Take a HTTP directory of Pubchem, download it's content (.gz in download_dir ) and extract the sdf files (in extracted_dir).
"""

from typing import List, Optional, Union
import os
import re
import requests
import gzip
import json
from urllib.request import urlretrieve
import shutil
import logging
import logging.config
from pathlib import Path
import concurrent.futures
from functools import partial

from rich.progress import track

logger = logging.getLogger(__name__)


def get_list_files(url: str) -> List[str]:
    response = requests.get(url)
    response.raise_for_status()
    # Use a regex pattern to find all href links in the page
    files = re.findall(r'href="([^"]+).md5"', response.text)

    return files


def download(url: str, path: str, md5: Optional[str] = None) -> str:
    """
    Download the md5 of the files
    Download a file from the specified url.
    Skip the downloading step if there exists a file satisfying the given MD5.

    Parameters:
        url (str): URL to download
        path (str, optional): path to store the downloaded file. If not specify tmp file.
        md5 (str, optional): MD5 of the file
    """

    if not os.path.exists(path) or compute_md5(path) != md5:
        logger.info("Downloading %s to %s" % (url, path))
        path, _ = urlretrieve(url, path)
    else:
        logger.info("Skipping %s since already downloaded at %s" % (url, path))

    return path


def retrieve_md5(url):
    md5_file, _ = urlretrieve(f"{url}.md5")
    with open(md5_file) as file:
        content = file.read()
    return content.split(" ")[0]


def check_md5_and_download(url: str, path: str):
    md5 = retrieve_md5(url)
    path = download(url, path, md5)
    return path


def compute_md5(file_name: str, chunk_size: int = 65536) -> str:
    """
    Compute MD5 of the file.

    Parameters:
        file_name (str): file name
        chunk_size (int, optional): chunk size for reading large files
    """
    import hashlib

    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()


def extract_file(archive_path: str, output_dir: os.PathLike[str]):
    # pass .sdf.gz to .sdf
    sdf_file_path = Path(output_dir) / Path(archive_path).with_suffix("").name
    if sdf_file_path.exists():
        logging.info(f"Skipping extraction of {sdf_file_path}, already exist")
        return
    logging.info(f"Extracting {archive_path} to {sdf_file_path}")
    # Decompress the .gz file and save the result as .sdf
    with gzip.open(archive_path, "rb") as f_in:
        with open(sdf_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_pubmed(output_dir: Union[str, os.PathLike[str]], num_worker: int):
    base_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"
    logger.info(f"Downloading PubMed from {base_url}")

    output_dir = Path(output_dir)
    download_dir = output_dir / "download_dir"
    extracted_dir = output_dir / "extracted_dir"

    download_dir.mkdir(parents=True, exist_ok=True)
    files = get_list_files(base_url)
    urls = [os.path.join(base_url, file) for file in files]
    paths = [os.path.join(download_dir, file) for file in files]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
        archive_paths = list(
            track(
                executor.map(check_md5_and_download, urls, paths),
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

    # ------

    output_dir = "/nasa/datasets/riken/projects/fundamental_models_202407/pubmed"
    num_worker = 5
    setup_logging(output_dir)
    download_pubmed(output_dir, num_worker)
    # -----
