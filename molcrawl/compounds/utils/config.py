from dataclasses import dataclass, field

from molcrawl.core.config import Config

from molcrawl.config.paths import COMPOUNDS_DIR, get_dataset_path


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
    data_preparation: Organix13PreparationConfig = field(default_factory=Organix13PreparationConfig)

    def __post_init__(self):
        if not isinstance(self.data_preparation, Organix13PreparationConfig):  # type: ignore[misc]
            self.data_preparation = Organix13PreparationConfig(**self.data_preparation)  # type: ignore[arg-type]
