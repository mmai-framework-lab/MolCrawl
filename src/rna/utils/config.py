from dataclasses import dataclass, field
from typing import Any, Dict, Union

from src.config.paths import RNA_DATASET_DIR
from core.config import Config


@dataclass
class CellxGenePreparationConfig:
    # Output directory where the preparation will be made
    output_dir: str = RNA_DATASET_DIR

    # Num of worker to use during parallel processing.
    num_worker: int = 8

    # Size of list of ids to give to each worker, save file will have `size_workload` number of ids in them.
    size_workload: int = 10000

    # Version of the CellxGene census
    census_version: str = "2023-12-15"

    # Filter condition to filter genes with few counts across a dataset.
    min_counts_genes: int = 2


@dataclass
class RnaConfig(Config):
    data_preparation: Union[CellxGenePreparationConfig, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.data_preparation, dict):
            self.data_preparation = CellxGenePreparationConfig(**self.data_preparation)
