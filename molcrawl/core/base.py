from __future__ import annotations

import json
import logging
import logging.config
import os
import sys
import uuid
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa
    import pyarrow.parquet as pq
from tqdm import tqdm


class TrainableTokenizer(ABC):
    def __init__(self):
        self.bulk_tokenizer_parquet = apply_fn_to_parqueet(self.tokenize_text)

    @abstractmethod
    def tokenize_text(self, text: str):
        pass

    def __len__(self):
        return len(self.tokenizer)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def train_new_from_iterator(self, iterator):
        super().train_new_from_iterator(iterator)


class UnTrainableTokenizer(ABC):
    def __init__(self):
        self.bulk_tokenizer_parquet = apply_fn_to_parqueet(self.tokenize_text)

    @abstractmethod
    def tokenize_text(self, text: str):
        pass


def read_parquet(file_path: str) -> "pq.ParquetFile":
    """
    Read parquet file and return as pandas DataFrame
    :param file_path: path to parquet file
    :return: pandas DataFrame
    """

    import pyarrow.parquet as pq

    return pq.read_table(file_path)


def save_parquet(table: "pa.Table", file_path: str):
    """
    Save a parquet file
    :param table: pyarrow Table
    :param file_path: path to save parquet file
    :return: None
    """

    import pyarrow.parquet as pq

    pq.write_table(table, file_path)


def split_table(table, chunk_size):
    num_rows = table.num_rows
    return [table.slice(offset, chunk_size) for offset in range(0, num_rows, chunk_size)]


def join_tables(chunks):
    import pyarrow as pa

    return pa.concat_tables(chunks)


def multiprocess_tokenization(func, table, column_name, new_column_name=None, processes=8):
    """
    Apply tokenization function to table using multiprocessing

    Args:
        func: Tokenization function to apply
        table: PyArrow table to process
        column_name: Name of column to tokenize
        new_column_name: Name for new tokenized column
        processes: Number of processes to use (default: 8)

    Returns:
        PyArrow table with tokenized column
    """
    split_tables = split_table(table, 10000)

    # Adjust processes if we have fewer chunks
    actual_processes = min(processes, len(split_tables))

    # Calculate chunksize for better load balancing
    # Using smaller chunksize for better responsiveness
    chunksize = max(1, len(split_tables) // (actual_processes * 4))

    logger = logging.getLogger(__name__)
    logger.info(f"Processing {len(split_tables)} chunks with {actual_processes} processes (chunksize={chunksize})")

    try:
        with Pool(processes=actual_processes) as pool:
            tokenized_tables = list(
                pool.imap(
                    partial(func, column_name=column_name, new_column_name=new_column_name),
                    tqdm(split_tables, total=len(split_tables)),
                    chunksize=chunksize,
                )
            )
    except Exception as e:
        logger.error(f"Error during multiprocess tokenization: {e}")
        raise

    return join_tables(tokenized_tables)


def apply_fn_to_parqueet(func):
    def inner(table, column_name, new_column_name=None):
        column_to_modify = table[column_name]

        import pyarrow as pa

        modified_column = pa.array([func(x.as_py()) for x in column_to_modify])

        if new_column_name is None:
            return table.set_column(table.column_names.index(column_name), column_name, modified_column)
        else:
            return table.append_column(new_column_name, modified_column)

    inner.__name__ = inner.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[inner.__module__], inner.__name__, inner)
    return inner


def setup_logging(output_dir: str, logging_config: str = "assets/logging_config.json"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(logging_config, "r") as file:
        config = json.load(file)
    logging_file = f"{output_dir}/logging.log"
    config["handlers"]["file"]["filename"] = logging_file
    if os.path.exists(logging_file):
        os.remove(logging_file)
    logging.config.dictConfig(config=config)
