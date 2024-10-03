from dataclasses import dataclass, field
from utils.config import Config


@dataclass
class CellxGenePreparationConfig:

    # Output directory where the preparation will be made
    output_dir: str

    # Num of worker to use during parallel processing.
    num_worker: int = 8

    # Size of list of ids to give to each worker, save file will have `size_workload` number of ids in them.
    size_workload: int = 10000


@dataclass
class RnaConfig(Config):

    data_preparation: CellxGenePreparationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.data_preparation = CellxGenePreparationConfig(**self.data_preparation)
