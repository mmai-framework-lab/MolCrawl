from dataclasses import dataclass, field

from molcrawl.core.config import Config


@dataclass
class MoleculeNLPreparationConfig(Config):
    # Path to save raw data
    dataset: str = "molecule_nat_lang/osunlp/SMolInstruct"

    # Path to save the processed and tokenized dataset
    save_path: str = "molecule_nat_lang/molecule_related_natural_language_tokenized.parquet"

    # Num of workers to use in the data preparation
    num_workers: int = 12


@dataclass
class MoleculeNLConfig(Config):
    data_preparation: MoleculeNLPreparationConfig = field(default_factory=MoleculeNLPreparationConfig)

    def __post_init__(self):
        if not isinstance(self.data_preparation, MoleculeNLPreparationConfig):  # type: ignore[misc]
            self.data_preparation = MoleculeNLPreparationConfig(**self.data_preparation)  # type: ignore[arg-type]
