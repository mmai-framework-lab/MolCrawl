from dataclasses import dataclass, field
from utils.config import Config


@dataclass
class UniProtPreparationConfig:
    # Which uniprot dataset to download must be one of the following:
    # "UniprotKB_reviewed", "UniprotKB_unreviewed", "UniRef100", "UniRef90", "UniRef50", "UniParc"
    dataset: str
    # Output directory where the preparation will be made
    output_dir: str
    # Special case for Uniparc download, num of worker to use.
    num_worker: int = 4


@dataclass
class ProteinSequenceConfig(Config):

    data_preparation: UniProtPreparationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.data_preparation = UniProtPreparationConfig(**self.data_preparation)
