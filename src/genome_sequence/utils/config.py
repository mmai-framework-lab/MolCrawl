from dataclasses import dataclass, field
from core.config import Config


@dataclass
class RefSeqPreparationConfig:

    # Output directory where the preparation will be made
    output_dir: str
    # Path to a directory containing one file per species to download from refseq (see assets/genome_species_list/species for example)
    # Possible groups are archaea, bacteria, fungi, invertebrate, metagenomes, plant, protozoa, vertebrate_mammalian, vertebrate_other, viral.
    path_species: str
    # Num of parallel worker to use, note that for download the worker are capped to 3
    num_worker: int = 16

    max_lines_per_file: int = 10000

    # Size of the vocabulary of the BPE tokenizer
    vocab_size: int = 4096


@dataclass
class GenomeSequenceConfig(Config):

    data_preparation: RefSeqPreparationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.data_preparation = RefSeqPreparationConfig(**self.data_preparation)
