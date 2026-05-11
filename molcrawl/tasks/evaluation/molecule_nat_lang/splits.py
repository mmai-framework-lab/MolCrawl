"""molecule_nat_lang pair loader uses a single-file split."""

from typing import Tuple

import pandas as pd


def split_by_column(
    df: pd.DataFrame, column: str, train_value: str = "train", test_value: str = "test"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if column not in df.columns:
        return df, df.iloc[0:0]
    train_df = df[df[column] == train_value].reset_index(drop=True)
    test_df = df[df[column] == test_value].reset_index(drop=True)
    return train_df, test_df
