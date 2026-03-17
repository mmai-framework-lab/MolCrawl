from dataclasses import dataclass, field

from molcrawl.config.paths import RNA_DATASET_DIR
from molcrawl.core.config import Config


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

    # Sqrt-scaling factor C for per-tissue cell subsampling.
    # Each tissue retains min(N, C * sqrt(N)) cells drawn without replacement.
    # Set to 0 (or omit) to disable subsampling and use all cells.
    sqrt_scale_factor: float = 0


@dataclass
class RnaConfig(Config):
    data_preparation: CellxGenePreparationConfig = field(default_factory=CellxGenePreparationConfig)

    def __post_init__(self):
        if not isinstance(self.data_preparation, CellxGenePreparationConfig):  # type: ignore[misc]
            self.data_preparation = CellxGenePreparationConfig(**self.data_preparation)  # type: ignore[arg-type]
