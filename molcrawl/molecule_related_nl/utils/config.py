from dataclasses import dataclass, field
from typing import Any, Dict, Union

from molcrawl.core.config import Config


@dataclass
class MoleculeNLPreparationConfig(Config):
    # Path to save raw data
    dataset: str = "molecule_nl/osunlp/SMolInstruct"

    # Path to save the processed and tokenized dataset
    save_path: str = "molecule_nl/molecule_related_natural_language_tokenized.parquet"

    # Num of workers to use in the data preparation
    num_workers: int = 12


@dataclass
class MoleculeNLConfig(Config):
    data_preparation: Union[MoleculeNLPreparationConfig, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.data_preparation, dict):
            self.data_preparation = MoleculeNLPreparationConfig(**self.data_preparation)
