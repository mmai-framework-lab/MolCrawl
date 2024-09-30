from typing import List
import concurrent.futures
import subprocess
from pathlib import Path
import logging
from functools import partial

from tqdm import tqdm

import gzip

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def uncompress(compress_file_path: Path, output_path: Path):
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with gzip.open(compress_file_path, "rt") as gzipped_file, open(
        output_path, "w"
    ) as smi_file:
        # Read from the .smi.gz and write to the .smi
        smi_file.write(gzipped_file.read())


def download_file(command: str, download_dir: Path, extract_dir: Path):
    result = subprocess.run(
        command,
        cwd=str(download_dir),
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logging.error(result.stderr.decode())
        return
    input_path = download_dir / command.split("-O")[-1].strip()
    output_path = extract_dir / command.split("-O")[-1].strip()
    output_path = output_path.with_suffix("")
    uncompress(input_path, output_path)


def remove_processed_files(commands: List[str], extract_dir: Path):
    remaining_commands = []
    for command in commands:
        output_path = extract_dir / command.split("-O")[-1].strip()
        output_path = output_path.with_suffix("")
        if not output_path.exists():
            remaining_commands.append(command)
    return remaining_commands


if __name__ == "__main__":
    uri_path = "Zinc/ZINC22-downloader-2D-smi.gz.wget"

    base_path = Path(
        "/nasa/datasets/riken/projects/fundamental_models_202407/zinc/zinc22"
    )
    # base_path = Path("Zinc/")
    download_dir = base_path / "download_dir"
    extract_dir = base_path / "extracted_dir"

    base_path.mkdir(exist_ok=True, parents=True)
    download_dir.mkdir(exist_ok=True)
    extract_dir.mkdir(exist_ok=True)

    with open(uri_path, "r") as f:
        commands = f.readlines()

    commands = remove_processed_files(commands, extract_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        func = partial(
            download_file, download_dir=download_dir, extract_dir=extract_dir
        )
        list(tqdm(executor.map(func, commands), total=len(commands)))
