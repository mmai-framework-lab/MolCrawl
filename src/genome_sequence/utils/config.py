from dataclasses import dataclass, field
from utils.config import Config


@dataclass
class RefSeqPreparationConfig:

    # Output directory where the preparation will be made
    output_dir: str
    # Num of parallel worker to use, note that for download the worker are capped to 3
    num_worker: int = 3


@dataclass
class GenomeSequenceConfig(Config):

    data_preparation: RefSeqPreparationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.data_preparation = RefSeqPreparationConfig(**self.data_preparation)
