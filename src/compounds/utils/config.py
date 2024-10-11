from dataclasses import dataclass, field
from utils.config import Config


@dataclass
class Organix13PreparationConfig:
    # Path to save the OrganiX13 dataset once is downloaded and processed by the script
    organix13_dataset: str

    # Path to save the processed and tokenized dataset
    save_path: str

    # Path to the vocabulary
    vocab_path: str

    # Max length of the tokenized sequences
    max_length: int = 256

    # Location to save raw unprocessed datasets
    raw_data_path: str = "src/compounds/dataset/organix13/raw"

@dataclass
class CompoundConfig(Config):

    data_preparation: Organix13PreparationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.data_preparation = Organix13PreparationConfig(**self.data_preparation)
