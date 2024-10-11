import json
import logging
import os
from pathlib import Path

from compounds.dataset.organix13.zinc.download_and_convert_to_parquet import download_zinc_files, convert_zinc_to_parquet
from compounds.dataset.organix13.combine_all import combine_all
from compounds.dataset.organix13.opv.prepare_opv import OPV
from compounds.dataset.organix13.download import download_datasets_from_repo


def setup_logging(output_dir: str, logging_config: str = "assets/logging_config.json"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(logging_config, "r") as file:
        config = json.load(file)
    logging_file = f"{output_dir}/logging.log"
    config["handlers"]["file"]["filename"] = logging_file
    if os.path.exists(logging_file):
        os.remove(logging_file)
    logging.config.dictConfig(config=config)

def download_datasets(raw_data_dir: str, output_dir: str):
    download_zinc_files()
    convert_zinc_to_parquet(raw_data_dir)
    OPV(raw_data_dir)
    download_datasets_from_repo(raw_data_dir)
    combine_all(raw_data_dir, output_dir)