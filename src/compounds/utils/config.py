from dataclasses import dataclass, field
from typing import Any, Dict, Union

from core.config import Config

from src.config.paths import COMPOUNDS_DIR, get_dataset_path


@dataclass
class Organix13PreparationConfig:
    # Path to save the OrganiX13 dataset once is downloaded and processed by the script
    organix13_dataset: str = COMPOUNDS_DIR + "/organix13"

    # Path to save the processed and tokenized dataset
    save_path: str = field(default_factory=lambda: get_dataset_path("compounds", "organix13_tokenized.parquet"))

    # Path to the vocabulary
    vocab_path: str = "assets/molecules/vocab.txt"

    # Max length of the tokenized sequences
    max_length: int = 256

    # Location to save raw unprocessed datasets
    raw_data_path: str = COMPOUNDS_DIR


@dataclass
class CompoundConfig(Config):
    data_preparation: Union[Organix13PreparationConfig, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.data_preparation, dict):
            self.data_preparation = Organix13PreparationConfig(**self.data_preparation)
