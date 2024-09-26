from abc import ABC, abstractmethod
import pyarrow.parquet as pq
import pyarrow as pa


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

    parquet_file = pq.ParquetFile(file_path)

    table = parquet_file.read_row_group(0)

    return table


def save_parquet(table: pa.Table, file_path: str):
    """
    Save a parquet file
    :param table: pyarrow Table
    :param file_path: path to save parquet file
    :return: None
    """

    pq.write_table(table, file_path)


def apply_fn_to_parqueet(func):

    def inner(table, column_name, new_column_name=None):
        column_to_modify = table[column_name]

        modified_column = pa.array([func(x.as_py()) for x in column_to_modify])

        return table.set_column(
            table.column_names.index(column_name),
            column_name if new_column_name is None else new_column_name,
            modified_column
        )

    return inner
