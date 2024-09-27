from argparse import ArgumentParser, Namespace
import os
import json
import logging
import logging.config
from pathlib import Path

from src.utils.base import read_parquet, save_parquet, multiprocess_tokenization
from src.molecules.tokenizer_utilities import MoleculesTokenizer


logger = logging.getLogger(__name__)


def get_script_arguments() -> Namespace:
    """
    Get the script arguments.

    :returns: The script arguments.
    """

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "-o13",
        "--organix13-dataset",
        type=str,
        required=True,
        help="Path to the root folder of the organix13 dataset."
    )

    argument_parser.add_argument(
        "-sp",
        "--save-path",
        type=str,
        required=True,
        help="Path to save the processed and tokenized dataset."
    )

    argument_parser.add_argument(
        "-vp",
        "--vocab-path",
        type=str,
        required=True,
        help="Path to the vocabulary."
    )
    
    argument_parser.add_argument(
        "-ml",
        "--max-length",
        type=str,
        required=False,
        help="Max length of the tokenized sequences.",
        default=256
    )

    return argument_parser.parse_args()

def setup_logging(output_dir: str, logging_config: str = "assets/logging_config.json"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(logging_config, "r") as file:
        config = json.load(file)
    logging_file = f"{output_dir}/logging.log"
    config["handlers"]["file"]["filename"] = logging_file
    if os.path.exists(logging_file):
        os.remove(logging_file)
    logging.config.dictConfig(config=config)


if __name__ == "__main__":
    script_arguments = get_script_arguments()
    setup_logging(Path(script_arguments.save_path).parent)

    try:

        organix13_dataset = read_parquet(
            file_path=script_arguments.organix13_dataset
        )

        tokenizer = MoleculesTokenizer(
            script_arguments.vocab_path,
            script_arguments.max_length,
        )

        logger.info(msg="Tokenizing...")
        processed_organix13 = multiprocess_tokenization(tokenizer.bulk_tokenizer_parquet, organix13_dataset, column_name="smiles", new_column_name="tokens")
        logger.info(msg="Tokenizing done.")

        logger.info(
            msg="Saving processed dataset to {}.".format(script_arguments.save_path)
        )

        save_parquet(
            table=processed_organix13,
            file_path=script_arguments.save_path
        )

    except Exception as exception_handle:
        logger.error(
            msg=exception_handle
        )

        raise
