from dataclasses import dataclass, field
from typing import Optional

from molcrawl.core.config import Config


@dataclass
class MoleculeNLPreparationConfig(Config):
    # Path to save raw data
    dataset: str = "molecule_nat_lang/osunlp/SMolInstruct"

    # Path to save the processed and tokenized dataset
    save_path: str = "molecule_nat_lang/molecule_related_natural_language_tokenized.parquet"

    # Num of workers to use in the data preparation
    num_workers: int = 12

    # Total number of examples (across all splits) to feed into the
    # GPT-2-style binarization step. ``None`` means "use every example
    # from each split"; a positive int draws an 80/10/10 random split of
    # exactly that many rows (legacy behaviour was a hard-coded 50000).
    number_sample: Optional[int] = None

    # Context length used when chunking concatenated token streams.
    context_length: int = 1024


@dataclass
class MoleculeNLConfig(Config):
    data_preparation: MoleculeNLPreparationConfig = field(default_factory=MoleculeNLPreparationConfig)

    def __post_init__(self):
        if not isinstance(self.data_preparation, MoleculeNLPreparationConfig):  # type: ignore[misc]
            self.data_preparation = MoleculeNLPreparationConfig(**self.data_preparation)  # type: ignore[arg-type]
