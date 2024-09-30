import subprocess
from pathlib import Path
import concurrent.futures
import logging
from functools import partial

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_file(line: str, download_dir):
    result = subprocess.run(
        ["wget", "-P", download_dir, line.strip()],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logging.error(result.stderr.decode())


if __name__ == "__main":
    base_path = Path("/nasa/datasets/riken/projects/fundamental_models_202407/Zinc")
    uri_path = base_path / "ZINC15-downloader-2D-txt.uri"
    download_dir = base_path / "download_dir"

    download_dir.mkdir(exist_ok=True)

    with open(uri_path, "r") as file:
        lines = file.readlines()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        func = partial(download_file, download_dir=download_dir)
        list(tqdm(executor.map(func, lines), total=len(lines)))
