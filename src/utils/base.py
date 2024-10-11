from abc import ABC, abstractmethod
import pyarrow.parquet as pq
import pyarrow as pa
import uuid
import sys

from functools import partial

from multiprocessing import Pool
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map


class TrainableTokenizer(ABC):

    def __init__(self):
        self.bulk_tokenizer_parquet = apply_fn_to_parqueet(self.tokenize_text)

    @abstractmethod
    def tokenize_text(self):
        pass

    @abstractmethod
    def train(self):
        pass


class UnTrainableTokenizer(ABC):

    def __init__(self):
        self.bulk_tokenizer_parquet = apply_fn_to_parqueet(self.tokenize_text)

    @abstractmethod
    def tokenize_text(self, text: str):
        pass


def read_parquet(file_path: str) -> pq.ParquetFile:
    """
    Read parquet file and return as pandas DataFrame
    :param file_path: path to parquet file
    :return: pandas DataFrame
    """

    return pq.read_table(file_path)


def save_parquet(table: pa.Table, file_path: str):
    """
    Save a parquet file
    :param table: pyarrow Table
    :param file_path: path to save parquet file
    :return: None
    """

    pq.write_table(table, file_path)


def split_table(table, chunk_size):
    num_rows = table.num_rows
    return [table.slice(offset, chunk_size) for offset in range(0, num_rows, chunk_size)]


def join_tables(chunks):
    return pa.concat_tables(chunks)


def multiprocess_tokenization(func, table, column_name, new_column_name=None, processes=24):
    split_tables = split_table(table, 10000)
    chunksize = len(split_tables) // processes if len(split_tables) // processes > 0 else 1
    
    with Pool(processes) as pool:
        tokenized_tables = [t for t in pool.map(
            partial(func, column_name=column_name, new_column_name=new_column_name),
            tqdm(split_tables, total=len(split_tables)),
            chunksize=chunksize
        )]

    return join_tables(tokenized_tables)


def apply_fn_to_parqueet(func):

    def inner(table, column_name, new_column_name=None):
        column_to_modify = table[column_name]

        modified_column = pa.array([func(x.as_py()) for x in column_to_modify])

        return table.set_column(
            table.column_names.index(column_name),
            column_name if new_column_name is None else new_column_name,
            modified_column
        )

    inner.__name__ = inner.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[inner.__module__], inner.__name__, inner)
    return inner
