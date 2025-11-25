from dataclasses import dataclass, field
from core.config import Config
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config.paths import get_dataset_path

from config.paths import COMPOUNDS_DIR


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
    raw_data_path: str = "src/compounds/dataset/organix13/raw"


@dataclass
class CompoundConfig(Config):
    data_preparation: Organix13PreparationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.data_preparation = Organix13PreparationConfig(**self.data_preparation)
