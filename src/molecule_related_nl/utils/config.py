from dataclasses import dataclass, field
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config.paths import get_dataset_path
from core.config import Config


@dataclass
class MoleculeNLPreparationConfig(Config):
    # Path to save raw data
    dataset: str = "src/molecule_related_nl/assets/raw_data"

    # Path to save the processed and tokenized dataset
    save_path: str = get_dataset_path("molecule_nl", "molecule_related_natural_language_tokenized.parquet")

    # Num of workers to use in the data preparation
    num_workers: int = 12


@dataclass
class MoleculeNLConfig(Config):
    data_preparation: MoleculeNLPreparationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.data_preparation = MoleculeNLPreparationConfig(**self.data_preparation)
