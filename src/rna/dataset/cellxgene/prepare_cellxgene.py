from argparse import ArgumentParser

from script.build_list import build_list
from script.download import download
from script.conv import convert
from rna.utils.config import RnaConfig


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    build_list(cfg.output_dir)
    download(cfg.output_dir, cfg.num_worker, cfg.size_workload)
    convert(cfg.output_dir, cfg.num_worker)
