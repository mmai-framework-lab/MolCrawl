from dataclasses import dataclass, field
from typing import Any, Dict, Union

from molcrawl.config.paths import PROTEIN_SEQUENCE_DIR
from molcrawl.core.config import Config


@dataclass
class UniProtPreparationConfig:
    # Which uniprot dataset to download must be one of the following:
    # "UniprotKB_reviewed", "UniprotKB_unreviewed", "UniRef100", "UniRef90", "UniRef50", "UniParc"
    dataset: str = "UniRef50"
    # Output directory where the preparation will be made
    output_dir: str = PROTEIN_SEQUENCE_DIR

    # If True use md5 to check if a file needs to be downloaded again, using md5
    # is very time consuming for large file. Otherwise we only check if the path exists.
    use_md5: bool = False

    # Special case for Uniparc download, num of worker to use.
    num_worker: int = 4

    # Number of sequence per files for raw files and parquet. It also reflex the number
    # of sequence loaded in memory during the processing of those files.
    max_lines_per_file: int = 10**6


@dataclass
class ProteinSequenceConfig(Config):
    data_preparation: Union[UniProtPreparationConfig, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.data_preparation, dict):
            self.data_preparation = UniProtPreparationConfig(**self.data_preparation)
