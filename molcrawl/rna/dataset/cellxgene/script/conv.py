import os
import concurrent.futures
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

from rich.progress import track

from molcrawl.rna.utils.config import RnaConfig


def get_file_to_process(output_dir: Path):
    with open(output_dir / "tissue_list.tsv", "r") as file:
        tissue_list = file.read().splitlines()

    arg_list = []
    for tissue in tissue_list:
        for filename in output_dir.glob(f"download_dir/{tissue}.*.jbl"):
            arg_list.append(filename)

    return arg_list


def run(filename, output_dir: Path):
    import joblib

    name = Path(filename).stem
    obj = joblib.load(filename)
    obj.write_h5ad(output_dir / f"extract/{name}.h5ad", compression="gzip")
    # obj.write_h5ad(filename,compression="lzf")


def convert(output_dir, num_worker):
    output_dir = Path(output_dir)
    arg_list = get_file_to_process(output_dir)

    os.makedirs(output_dir / "extract/", exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
        func = partial(run, output_dir=output_dir)
        list(
            track(
                executor.map(func, arg_list),
                description="Conversion...",
                total=len(arg_list),
            )
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation
    convert(cfg.output_dir, cfg.num_worker)
