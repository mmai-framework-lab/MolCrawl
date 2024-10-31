from argparse import ArgumentParser
import os
import logging
import logging.config

from pathlib import Path

from core.base import read_parquet, save_parquet, multiprocess_tokenization, setup_logging
from compounds.utils.tokenizer import CompoundsTokenizer, ScaffoldsTokenizer
from compounds.utils.config import CompoundConfig
from compounds.utils.general import download_datasets


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = CompoundConfig.from_file(args.config).data_preparation

    setup_logging(Path(cfg.save_path).parent / "compounds_logs")

    os.path.exists(cfg.raw_data_path) or os.makedirs(cfg.raw_data_path)
    download_datasets(cfg.raw_data_path, cfg.organix13_dataset)

    organix13_dataset = read_parquet(file_path=os.path.join(cfg.organix13_dataset, "OrganiX13.parquet"))

    mol_tokenizer = CompoundsTokenizer(
        cfg.vocab_path,
        cfg.max_length,
    )

    logger.info(msg="Tokenizing SMILES...")
    processed_organix13 = multiprocess_tokenization(
        mol_tokenizer.bulk_tokenizer_parquet, organix13_dataset, column_name="smiles", new_column_name="tokens"
    )

    scaffolds_tokenizer = ScaffoldsTokenizer(
        cfg.vocab_path,
        cfg.max_length,
    )

    logger.info(msg="Tokenizing Scaffolds...")
    processed_organix13 = multiprocess_tokenization(
        scaffolds_tokenizer.bulk_tokenizer_parquet,
        processed_organix13,
        column_name="smiles",
        new_column_name="scaffold_tokens",
    )

    logger.info(msg="Tokenizing done.")

    logger.info(msg="Saving processed dataset to {}.".format(cfg.save_path))

    save_parquet(table=processed_organix13, file_path=cfg.save_path)
