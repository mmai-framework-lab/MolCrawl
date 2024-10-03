# See https://www.uniprot.org/help/downloads
from typing import Dict, Optional, Union
import json
import os
import logging
import logging.config
from pathlib import Path
import gzip
from urllib.request import urlretrieve
import xml.etree.ElementTree as ET
import concurrent.futures
import hashlib
from argparse import ArgumentParser

from rich.progress import track

from protein_sequence.utils.configs import ProteinSequenceConfig

logger = logging.getLogger(__name__)


class UniProtDatasetEnum:
    UniprotKB_reviewed = "UniprotKB_reviewed"
    UniprotKB_unreviewed = "UniprotKB_unreviewed"
    UniRef100 = "UniRef100"
    UniRef90 = "UniRef90"
    UniRef50 = "UniRef50"
    UniParc = "UniParc"


pubmed_fasta_url = {
    UniProtDatasetEnum.UniprotKB_reviewed: "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
    UniProtDatasetEnum.UniprotKB_unreviewed: "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz",
    UniProtDatasetEnum.UniRef100: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz",
    UniProtDatasetEnum.UniRef90: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz",
    UniProtDatasetEnum.UniRef50: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
    UniProtDatasetEnum.UniParc: "https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniparc/fasta/active/",
}

pubmed_fasta_md5 = {
    UniProtDatasetEnum.UniprotKB_reviewed: "a867f9a037febd55ab00d788294439f9",
    UniProtDatasetEnum.UniprotKB_unreviewed: "d79a23ccdde970c705f71af9a1b906c3",
    UniProtDatasetEnum.UniRef100: "e220bcd8ad33a6f44d5ecb5c794d7a46",
    UniProtDatasetEnum.UniRef90: "76c10e91637637adfd1bed2d82adc914",
    UniProtDatasetEnum.UniRef50: "53b922d6802c8616e7d77b616ad01708",
}


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


def compute_md5(file_name: str, chunk_size: int = 65536) -> str:
    """
    Compute MD5 of the file.

    Parameters:
        file_name (str): file name
        chunk_size (int, optional): chunk size for reading large files
    """

    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        while chunk := fin.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def unzip_file(archive_path: str, output_path: Path):
    if output_path.exists():
        logger.info(f"Skipping extraction, {output_path} already exist")
        return
    logger.info(f"Extracting {output_path}")
    with gzip.open(archive_path, "rt") as archive, open(output_path, "w") as file:
        file.write(archive.read())


def download_full_http_dir(
    url: str, download_dir: Union[str, os.PathLike[str]], num_worker: int
):
    file_md5_dict = get_url_md5_mapping(url)
    urls = [os.path.join(url, file) for file in file_md5_dict.keys()]
    paths = [os.path.join(download_dir, file) for file in file_md5_dict.keys()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
        archive_paths = list(
            track(
                executor.map(download, urls, paths, file_md5_dict.values()),
                total=len(urls),
                description="Downloading...",
            )
        )
    return archive_paths


def get_url_md5_mapping(url: str) -> Dict:
    file_path, _ = urlretrieve(os.path.join(url, "RELEASE.metalink"))

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define the namespace (from the XML structure)
    namespace = {"ns": "http://www.metalinker.org/"}

    file_md5_dict = {}

    # Iterate through the XML structure to find file names and md5 hashes
    for file in root.findall(".//ns:file", namespace):
        file_name = file.attrib.get("name")
        md5_hash = file.find('.//ns:hash[@type="md5"]', namespace)
        if md5_hash is not None:
            md5_hash = md5_hash.text
        if file_name and md5_hash:
            file_md5_dict[file_name] = md5_hash

    return file_md5_dict


def process_dataset(
    dataset: str, output_dir: Union[str, os.PathLike[str]], num_worker: int
):
    logging.info(f"Processing dataset {dataset}...")
    logging.info("Downloading archive from the server...")
    if dataset == UniProtDatasetEnum.UniParc:
        output_dir = Path(output_dir) / "uniparc"
        download_dir = output_dir / "archive"
        download_dir.mkdir(parents=True, exist_ok=True)
        archive_path = download_full_http_dir(
            pubmed_fasta_url[dataset], download_dir, num_worker
        )

        paths = [output_dir / Path(path).with_suffix("").name for path in archive_path]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
            list(
                track(
                    executor.map(unzip_file, archive_path, paths),
                    total=len(paths),
                    description="Extracting...",
                )
            )
    else:
        download_path = Path(output_dir) / Path(pubmed_fasta_url[dataset]).name
        download_path = download(
            pubmed_fasta_url[dataset], str(download_path), md5=pubmed_fasta_md5[dataset]
        )
        logging.info("Decompressing the archive...")
        unzip_file(
            download_path, Path(output_dir) / Path(download_path).with_suffix("").name
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
    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation
    setup_logging(cfg.output_dir)
    process_dataset(cfg.dataset, cfg.output_dir, cfg.num_worker)
