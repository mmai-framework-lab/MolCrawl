from argparse import ArgumentParser
import os
import logging
import logging.config
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from core.base import setup_logging
from molecule_related_nl.dataset.download import download_hf_dataset
from molecule_related_nl.utils.config import MoleculeNLConfig
from molecule_related_nl.utils.general import read_dataset, save_dataset

from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer


logger = logging.getLogger(__name__)


def run_statistics(series, column_name):
    series_length = [len(i) for i in series]
    plt.hist(series_length, bins=np.arange(0, 200, 1))
    plt.xlabel("Length of tokenized {}".format(column_name))
    plt.title("Distribution of tokenized {} lengths".format(column_name))
    plt.savefig("assets/img/molecule_nl_tokenized_{}_lengths_dist.png".format(column_name))
    plt.close()
    logger.info(msg="Saved distribution of tokenized {} lengths to assets/img/compounds_tokenized_{}_lengths_dist.png".format(column_name, column_name))


def calculate_statistics(dataset, split):
    inp_out = [i+j for i,j in zip(dataset[split]["input_ids"], dataset[split]["output_ids"])]
    num_samples = len(inp_out)
    num_tokens = sum(len(i) for i in inp_out)

    run_statistics(inp_out, split)
    return num_samples, num_tokens


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = MoleculeNLConfig.from_file(args.config).data_preparation

    logging_dir = Path(cfg.save_path).parent / "molecule_related_natural_language_logs"
    os.path.exists(logging_dir) or os.makedirs(logging_dir)
    setup_logging(logging_dir)

    os.path.exists(cfg.dataset) or os.makedirs(cfg.dataset)

    logger.info(msg="Downloading Dataset...")
    try:
        download_hf_dataset(cfg.dataset)
    except Exception as e:
        logger.error(msg="Failed to download dataset. This error often occurs as you have already downloaded the dataset.")
        logger.error(msg=e)

    dataset = read_dataset(cfg.dataset)

    tokenizer = MoleculeNatLangTokenizer()

    logger.info(msg="Tokenizing Scaffolds...")

    processed_dataset = {}
    for split in dataset.keys():
        processed_dataset[split] = dataset[split].map(
            tokenizer.tokenize_dict,
            batched = False,
            num_proc = cfg.num_workers,
            load_from_cache_file = False,
            desc="Tokenizing {}".format(split),
        )

    logger.info(msg="Computing Dataset Statistics...")
    total_num_samples = 0
    total_num_tokens = 0
    for split in processed_dataset.keys():
        logger.info(msg=f"{split}")
        num_samples, num_tokens = calculate_statistics(processed_dataset, split)
        logger.info(msg=f"Number of examples: {num_samples}")
        logger.info(msg=f"Number of tokens: {num_tokens}")
        total_num_samples += num_samples
        total_num_tokens += num_tokens

    logger.info(msg="Total number of tokens: {}".format(total_num_samples))
    logger.info(msg="Total number of examples: {}".format(total_num_tokens))

    logger.info(msg="Saving processed dataset to {}.".format(cfg.save_path))
    save_dataset(processed_dataset, cfg.save_path)
